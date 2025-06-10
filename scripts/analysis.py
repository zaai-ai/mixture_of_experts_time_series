import pandas as pd

from utilsforecast.losses import smape, mape, rmae, mae, mase, rmse, rmsse
from neuralforecast.losses.numpy import smape as smape2

from modelradar.evaluate.radar import ModelRadar
from modelradar.visuals.plotter import ModelRadarPlotter, SpiderPlot
from pandas.tseries.offsets import DateOffset, MonthEnd, QuarterEnd, YearEnd
from functools import partial

results_list = {
    # 'm1m': 'results,gluonts,m1_monthly.csv',
    # 'm1q': 'results,gluonts,m1_quarterly.csv',
    # 'm1y': 'results,gluonts,m1_yearly.csv',
    # 'tm': 'results,gluonts,tourism_monthly.csv',
    # 'tq': 'results,gluonts,tourism_quarterly.csv',
    # 'ty': 'results,gluonts,tourism_yearly.csv',
    # 'm3m': 'results,m3,Monthly.csv',
    # 'm3q': 'results,m3,Quarterly.csv',
    # 'm3y': 'results,m3,Yearly.csv',
    # 'm4m': 'results,m4,Monthly.csv',
    'm4q': 'results,m4,Quarterly.csv',
    # 'm4y': 'results,m4,Yearly.csv',
}

all_results = []
start_date = pd.to_datetime('1994-01-31')

def calculate_smape(Y_test, predictions, model_name):
    y_true = Y_test['y'].values
    y_hat = predictions[model_name].values

    n_series = Y_test['unique_id'].nunique()
    y_true = y_true.reshape(n_series, -1)
    y_hat = y_hat.reshape(n_series, -1)

    smape_value = smape2(y_true, y_hat)
    return smape_value

def convert_to_date(group, freq):
    group = group.copy()
    periods = len(group)
    group['ds'] = pd.date_range(start=start_date, periods=periods, freq=freq)
    return group

for dataset, file_path in results_list.items():
    try:
        df = pd.read_csv(file_path)
        df['unique_id'] = dataset + '_' + df['unique_id']

        # print(df['ds'].head())
        # print(df['ds'].describe())
        # if pd.api.types.is_integer_dtype(df['ds']):

        #     df = df.groupby('unique_id', group_keys=False).apply(partial(convert_to_date, freq=dataset[-1].upper()+'E'))

        # print(df['ds'].head())

        all_results.append(df)
        print(f"Successfully loaded: {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Concatenate all dataframes
if all_results:
    combined_results = pd.concat(all_results, ignore_index=True)
    print(f"Combined {len(all_results)} files with {len(combined_results)} total rows")
else:
    combined_results = pd.DataFrame()
    print("No files were successfully loaded")

cv = combined_results.copy()


print(calculate_smape(cv, cv, "SeasonalNaive"))
print("HEL")
raise Exception()
# -----

# pick hard uids based on smape

radar = ModelRadar(cv_df=cv,
                   metrics=[smape],
                   model_names=['SeasonalNaive'],
                   hardness_reference='SeasonalNaive',
                   ratios_reference='NBEATS',
                   cvar_quantile=0.95,
                   hardness_quantile=0.95,
                   rope=10)

err = radar.evaluate(keep_uids=True)
print(err)
print(err.mean())
radar.uid_accuracy.get_hard_uids(err, return_df=False)
hard_uid_list = radar.uid_accuracy.hard_uid

# ------
from functools import partial

rmae_func = partial(rmae, baseline='SeasonalNaive')
mase_func = partial(mase, seasonality=1)



radar = ModelRadar(cv_df=cv,
                   # metrics=[smape, mae],
                   metrics=[smape],
                   # metrics=[rmae_func],
                   model_names=['NBeatsMoe', 'NBEATS', 'NBeatsStackMoe', 'SeasonalNaive'],
                   hardness_reference='SeasonalNaive',
                   ratios_reference='NBEATS',
                   cvar_quantile=0.95,
                   hardness_quantile=0.95,
                   rope=10)

err = radar.evaluate(keep_uids=True)
# Calculate relative performance compared to SeasonalNaive
# Calculate relative performance compared to SeasonalNaive in-place
# seasonal_naive_values = err['SeasonalNaive'].copy()
# for col in err.columns:
#     if col != 'SeasonalNaive':
#         err[col] = err[col] / seasonal_naive_values
#
# # Set SeasonalNaive column as reference (value of 1.0)
# err['SeasonalNaive'] = 1.0

err.head()
print('Overall scores')
print(err.mean())
# evaluate(df=test_extended,
#          models=['NBeatsMoe', 'NBEATS', 'NBeatsStackMoe'],
#          metrics=[seasonal_mase],
#          train_df=train)

# err_hard = radar.uid_accuracy.get_hard_uids(err)
err_hard = err.loc[hard_uid_list, :]

print('Overall scores on hard time series')
print(err_hard.mean())
# print(getattr(err, "mean")(axis=0))  # or any other argument like skipna=True


err_anomalies = radar.evaluate_by_anomaly(anomaly_col='is_anomaly', mode='observations')
# err_anomalies = radar.evaluate_by_anomaly(anomaly_col='is_anomaly', mode='series')

err_anomalies.head()

print('Scores by horizon bound')
print(radar.evaluate_by_horizon_bounds(return_plot=False))

eval_fhorizon = radar.evaluate_by_horizon()

print('Win/draw/loss ratios')
print(radar.rope.get_winning_ratios(err))

print('Win/draw/loss ratios on hard time series')
print(radar.rope.get_winning_ratios(err_hard))

print('Expected shortfall')
print(radar.uid_accuracy.expected_shortfall(err))

error_on_anomalies = radar.evaluate_by_group(group_col='anomaly_status')
print('By anomaly status')
print(error_on_anomalies)
cv['anomaly_status'].value_counts()
