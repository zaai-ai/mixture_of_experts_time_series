import pandas as pd
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast
from utils import load_dataset, train_test_split
from neuralforecast.losses.numpy import smape 

df = load_dataset('m4', {'directory': '.', 'group': 'Monthly'})
df, horizon, n_lags, freq_str, freq_int = load_dataset('gluonts_tourism_monthly', {})
print(df)
# horizon = 18
train, test = train_test_split(df, horizon)

sf = StatsForecast(models=[SeasonalNaive(season_length=12)], freq="ME", n_jobs=1)

sf.fit(df=train)


def calculate_smape(Y_test, predictions, model_name):
    y_true = Y_test['y'].values
    y_hat = predictions[model_name].values

    n_series = Y_test['unique_id'].nunique()
    y_true = y_true.reshape(n_series, -1)
    y_hat = y_hat.reshape(n_series, -1)

    smape_value = smape(y_true, y_hat)
    return smape_value

Y_pred = sf.predict(h=horizon)


print(f"SMAPE: {calculate_smape(test, Y_pred, 'SeasonalNaive')}")



