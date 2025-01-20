import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from models.SimpleMoe import SimpleMoe
from datasetsforecast.m3 import M3
from neuralforecast.losses.pytorch import MSE, SMAPE


def train_test_split(df: pd.DataFrame, horizon: int):
    df_by_unq = df.groupby('unique_id')

    train_l, test_l = [], []
    for _, df_ in df_by_unq:
        df_ = df_.sort_values('ds')

        train_df_g = df_.head(-horizon)
        test_df_g = df_.tail(horizon)

        train_l.append(train_df_g)
        test_l.append(test_df_g)

    train_df = pd.concat(train_l).reset_index(drop=True)
    test_df = pd.concat(test_l).reset_index(drop=True)

    return train_df, test_df

Y_ALL = M3.load(directory='./data/m3', group='Monthly')[0]

Y_train_df, Y_test_df = train_test_split(Y_ALL, horizon=12)

model = SimpleMoe(h=12,
             input_size=24,
             dropout=0.1,
             loss=SMAPE(),
             valid_loss=SMAPE(),
             early_stop_patience_steps=3,
             batch_size=32)

fcst = NeuralForecast(models=[model], freq='M')
fcst.fit(df=Y_train_df, static_df=None, val_size=12)
forecasts = fcst.predict(futr_df=Y_test_df)

# Plot predictions
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['RMoK'], c='blue', label='Forecast')
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Year', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()

