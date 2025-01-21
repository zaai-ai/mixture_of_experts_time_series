import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from utilsforecast.evaluation import evaluate
from neuralforecast.losses.numpy import smape
from functools import partial
from models.SimpleMoe import SimpleMoe
from datasetsforecast.m3 import M3
from neuralforecast.losses.pytorch import *
import hydra
from omegaconf import DictConfig

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

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    
    # Load dataset
    Y_ALL = M3.load(directory=cfg.data.m3_monthly.directory, group=cfg.data.m3_monthly.group)[0]
    Y_train_df, Y_test_df = train_test_split(Y_ALL, horizon=cfg.data.m3_monthly.horizon)

    # Initialize the model
    model = SimpleMoe(
        h=cfg.model.simple_moe.h,
        input_size=cfg.model.simple_moe.input_size,
        dropout=cfg.model.simple_moe.dropout,
        loss=eval(cfg.model.simple_moe.loss)(),
        valid_loss=eval(cfg.model.simple_moe.valid_loss)(),
        early_stop_patience_steps=cfg.model.simple_moe.early_stop_patience_steps,
        batch_size=cfg.model.simple_moe.batch_size
    )

    # Forecasting
    fcst = NeuralForecast(models=[model], freq=cfg.forecast.default_forecast.freq)
    fcst.fit(df=Y_train_df, static_df=None, val_size=cfg.data.m3_monthly.horizon)
    forecasts = fcst.predict(futr_df=Y_test_df)

    # Plot predictions
    fig, ax = plt.subplots(1, 1, figsize=cfg.plot.default_plot.figsize)
    Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id', 'ds'])
    plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
    plot_df = pd.concat([Y_train_df, plot_df])

    # Plot all predictions
    for unique_id in plot_df['unique_id'].unique()[-10::]:
        temp_df = plot_df[plot_df['unique_id'] == unique_id]
        plt.plot(temp_df['ds'], temp_df['y'], label=f'True {unique_id}')
        plt.plot(temp_df['ds'], temp_df['SimpleMoe'], label=f'Forecast {unique_id}')

    ax.set_title(cfg.plot.default_plot.title, fontsize=22)
    ax.set_ylabel(cfg.plot.default_plot.ylabel, fontsize=20)
    ax.set_xlabel(cfg.plot.default_plot.xlabel, fontsize=20)
    ax.legend(prop={'size': cfg.plot.default_plot.legend_font_size})
    ax.grid(cfg.plot.default_plot.grid)

    # Save the plot
    fig.savefig(cfg.plot.default_plot.save_path)
    plt.show()
    
    # evaluate
    y_true = Y_test_df['y'].values
    y_hat = Y_hat_df['SimpleMoe'].values
    
    n_series = Y_test_df['unique_id'].nunique()
    
    y_true = y_true.reshape(n_series, -1)
    y_hat = y_hat.reshape(n_series, -1)
    
    print(f'sMAPE: {smape(y_true, y_hat)}')
    

if __name__ == "__main__":
    main()
