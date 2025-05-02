from pprint import pprint
from functools import partial

import pandas as pd
import numpy as np

from neuralforecast import NeuralForecast

from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast
from neuralforecast.losses.numpy import mase, rmse, rmae
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape, rmae, mape

from modelradar.evaluate.radar import ModelRadar
from modelradar.visuals.plotter import ModelRadarPlotter, SpiderPlot


from utils import load_dataset, train_test_split

from models.NBeatsMoe import NBeatsMoe
from models.NBeatsStackMoe import NBeatsStackMoe
from neuralforecast.models import NBEATS

from models.config import config

dataset='gluonts'
group='tourism_quarterly'
# m1_yearly
# m1_quarterly
# tourism_monthly
# tourism_quarterly
# tourism_yearly
df, horizon, n_lags, freq_str, freq_int = load_dataset(f'{dataset}_{group}', {})
# df, horizon, n_lags, freq_str, freq_int = load_dataset(f'{dataset}_{group}', {})
# df, horizon, n_lags, freq_str, freq_int = load_dataset(f'{dataset}_{group}', {})
# df, horizon, n_lags, freq_str, freq_int = load_dataset(f'{dataset}_{group}', {})
# df = load_dataset(dataset, {'directory': '.', 'group': group})
# df = load_dataset(dataset, {'directory': '.', 'group': group})
# df = load_dataset(dataset, {'directory': '.', 'group': group})
# config['m3']['Yearly']

# horizon = 18
# horizon = 8
# horizon = 4
# n_lags = 18
# n_lags = 8
# n_lags = 2
# freq_int = 12
# freq_int = 4
# freq_int = 1
# freq_str = 'ME'
# freq_str = 'QE'
# freq_str = 'Y'
# freq_str=12

train, test = train_test_split(df, horizon)

train.to_csv(f'train,{dataset},{group}.csv', index=False)

models = [
    NBeatsMoe(h=horizon,
              # input_size=n_lags,
              # mlp_units=3 * [[256, 256]],
              # max_steps=500,
              accelerator='mps',
              **config[dataset][group]['NBeatsMoe']),
    NBEATS(h=horizon,
           # input_size=n_lags,
           # max_steps=500,
           accelerator='mps',
           **config[dataset][group]['NBEATS']),
    NBeatsStackMoe(h=horizon,
                   # input_size=n_lags,
                   # mlp_units=3 * [[128, 128]],
                   # max_steps=500,
                   accelerator='mps',
                   **config[dataset][group]['NBeatsStackMoe']),
]

# seasonal_mase = partial(mase, seasonality=freq_int)

stats_models = [SeasonalNaive(season_length=freq_int)]

sf = StatsForecast(models=stats_models, freq=freq_str, n_jobs=1)

fcst = NeuralForecast(models=models, freq=freq_str)
fcst.fit(df=train, static_df=None, val_size=horizon)
sf.fit(df=train)

Y_pred_df_sf = sf.predict(h=horizon, level=[99])
Y_pred_df = fcst.predict()

Y_pred_df = Y_pred_df.merge(Y_pred_df_sf, on=['unique_id', 'ds'])

test_extended = test.merge(Y_pred_df, on=['unique_id', 'ds'])

is_outside_pi = (test_extended['y'] >= test_extended['SeasonalNaive-hi-99']) | (
            test_extended['y'] <= test_extended['SeasonalNaive-lo-99'])
test_extended['is_anomaly'] = is_outside_pi.astype(int)
test_extended = test_extended.drop(columns=['SeasonalNaive-lo-99', 'SeasonalNaive-hi-99'])
test_extended['anomaly_status'] = test_extended['is_anomaly'].map({0: 'Non-anomalies', 1: 'Anomalies'})

test_extended.to_csv(f'results,{dataset},{group}.csv', index=False)

radar = ModelRadar(cv_df=test_extended,
                   metrics=[smape],
                   model_names=['NBeatsMoe', 'NBEATS', 'NBeatsStackMoe', 'SeasonalNaive'],
                   hardness_reference='SeasonalNaive',
                   ratios_reference='NBEATS',
                   cvar_quantile=0.85,
                   hardness_quantile=0.85,
                   rope=10)

err = radar.evaluate(keep_uids=True)

err.head()
print(radar.evaluate(keep_uids=False))
# evaluate(df=test_extended,
#          models=['NBeatsMoe', 'NBEATS', 'NBeatsStackMoe'],
#          metrics=[seasonal_mase],
#          train_df=train)

err_hard = radar.uid_accuracy.get_hard_uids(err)

print("err_hard.mean()")
print(err_hard.mean())

err_anomalies = radar.evaluate_by_anomaly(anomaly_col='is_anomaly', mode='observations')
# err_anomalies = radar.evaluate_by_anomaly(anomaly_col='is_anomaly', mode='series')

err_anomalies.head()

print(radar.evaluate_by_horizon_bounds(return_plot=False))

eval_fhorizon = radar.evaluate_by_horizon()

print(radar.rope.get_winning_ratios(err))

print(radar.rope.get_winning_ratios(err_hard))

print(radar.uid_accuracy.expected_shortfall(err))

error_on_anomalies = radar.evaluate_by_group(group_col='anomaly_status')
print(error_on_anomalies)
