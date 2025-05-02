from pprint import pprint
from functools import partial

import pandas as pd
import numpy as np

from neuralforecast import NeuralForecast

from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast
from neuralforecast.losses.numpy import mase
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape

from modelradar.evaluate.radar import ModelRadar
from modelradar.visuals.plotter import ModelRadarPlotter, SpiderPlot

from utils import load_dataset, train_test_split

from models.NBeatsMoe import NBeatsMoe
from models.NBeatsStackMoe import NBeatsStackMoe
from neuralforecast.models import NBEATS

# df, horizon, n_lags, freq_str, freq_int = load_dataset('gluonts_m1_yearly', {})
# df, horizon, n_lags, freq_str, freq_int = load_dataset('gluonts_m1_quarterly', {})
# df, horizon, n_lags, freq_str, freq_int = load_dataset('gluonts_tourism_monthly', {})
# df, horizon, n_lags, freq_str, freq_int = load_dataset('gluonts_tourism_quarterly', {})
df = load_dataset('m3', {'directory': '.', 'group': 'Monthly'})
# df = load_dataset('m3', {'directory': '.', 'group': 'Quarterly'})
# df = load_dataset('m4', {'directory': '.', 'group': 'Monthly'})
horizon = 18
# horizon = 8
n_lags = 18
# n_lags = 8
freq_int = 12
# freq_int = 4
freq_str = 'ME'
# freq_str = 'QE'
# freq_str=12

train, test = train_test_split(df, horizon)

models = [
    NBeatsMoe(h=horizon,
              input_size=n_lags,
              mlp_units=3 * [[256, 256]],
              max_steps=1000,
              accelerator='cpu'),
    NBEATS(h=horizon, input_size=n_lags,
           max_steps=1000,
           accelerator='cpu'),
    NBeatsStackMoe(h=horizon,
                   input_size=n_lags,
                   mlp_units=3 * [[128, 128]],
                   max_steps=1000,
                   accelerator='cpu'),
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

radar = ModelRadar(cv_df=test_extended,
                   metrics=[smape],
                   model_names=['NBeatsMoe', 'NBEATS', 'NBeatsStackMoe', 'SeasonalNaive'],
                   hardness_reference='SeasonalNaive',
                   ratios_reference='NBEATS',
                   cvar_quantile=0.85,
                   hardness_quantile=0.85,
                   rope=10)

err = radar.evaluate(keep_uids=True)

# NBeatsMoe         0.070113
# NBEATS            0.070131
# NBeatsStackMoe    0.071195
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
