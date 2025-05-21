from pprint import pprint
from functools import partial

import pandas as pd
import numpy as np

from neuralforecast import NeuralForecast

from neuralforecast.losses.numpy import smape, mase
from utilsforecast.evaluation import evaluate

from utils import load_dataset, train_test_split

from models.NBeatsMoe import NBeatsMoe
from models.NBeatsStackMoe import NBeatsStackMoe
from neuralforecast.models import NBEATS

# df, horizon, n_lags, freq_str, freq_int = load_dataset('gluonts_m1_monthly', {})
# df, horizon, n_lags, freq_str, freq_int = load_dataset('gluonts_tourism_monthly', {})
df = load_dataset('m3', {'directory': '.', 'group': 'Monthly'})
# df = load_dataset('m4', {'directory': '.', 'group': 'Monthly'})
horizon = 36
n_lags = 18
freq_int = 12
freq_str='ME'
# freq_str = 12

train, test = train_test_split(df, horizon)

models = [
    NBeatsMoe(h=horizon,
              input_size=n_lags,
              mlp_units=3 * [[256, 256]],
              max_steps=1000,
              accelerator='mps'),
    NBEATS(h=horizon, input_size=n_lags, max_steps=1000, accelerator='mps'),
    NBeatsStackMoe(h=horizon,
                   input_size=n_lags,
                   mlp_units=3 * [[128, 128]],
                   max_steps=1000,
                   accelerator='mps'),
]

seasonal_mase = partial(mase, seasonality=freq_int)

fcst = NeuralForecast(models=models, freq=freq_str)
fcst.fit(df=train, static_df=None, val_size=horizon)

Y_pred_df = fcst.predict()

test_extended = test.merge(Y_pred_df, on=['unique_id', 'ds'])

# evaluate(df=test_extended,
#          models=['NBeatsMoe', 'NBEATS', 'NBeatsStackMoe'],
#          metrics=[seasonal_mase],
#          train_df=train)

model_names = ['NBeatsMoe', 'NBEATS', 'NBeatsStackMoe']

# todo modelradar
model_scores = {}
for mod in model_names:
    test_grouped = test_extended.groupby('unique_id')

    scores_l = []
    for uid, test_group in test_grouped:
        train_uid = train[train['unique_id'] == uid]

        score = seasonal_mase(
            y=test_group['y'].values,
            y_hat=test_group[mod].values,
            y_train=train_uid['y'].values
        )
        # score = smape(
        #     y=test_group['y'].values,
        #     y_hat=test_group[mod].values,
        #     # y_train=train_uid['y'].values
        # )

        scores_l.append(score)

    model_scores[mod] = np.mean(scores_l)

pprint(model_scores)
