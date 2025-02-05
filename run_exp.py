import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.losses.numpy import smape
from datasetsforecast.m3 import M3
from neuralforecast.losses.pytorch import SMAPE
import hydra
from omegaconf import DictConfig

# Import your model classes (here we assume the module name matches the model name)
from models.SimpleMoe import SimpleMoe


def train_test_split(df: pd.DataFrame, horizon: int):
    df_by_unq = df.groupby('unique_id')
    train_list, test_list = [], []
    for _, df_ in df_by_unq:
        df_ = df_.sort_values('ds')
        train_list.append(df_.head(-horizon))
        test_list.append(df_.tail(horizon))
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df, test_df


def get_config_value(cfg_val, idx):
    """
    Helper: If cfg_val is a list then return the idx-th element; otherwise return cfg_val.
    """
    if isinstance(cfg_val, list):
        return cfg_val[idx % len(cfg_val)]
    return cfg_val


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):

    # Loop over each active dataset specified in the config
    for dataset_name in cfg.active_datasets:
        print(f"\n==== Running experiment for dataset: {dataset_name} ====")
        dataset_cfg = cfg.data.get(dataset_name)
        if dataset_cfg is None:
            print(f"Dataset configuration '{dataset_name}' not found in config.data!")
            continue

        # Load the dataset. Here we assume that for each dataset you know how to load it.
        # For instance, if dataset_name is "m3_monthly", we load it with M3.load.
        if dataset_name == "m3_monthly":
            Y_ALL = M3.load(
                directory=dataset_cfg.directory,
                group=dataset_cfg.group
            )[0]
        else:
            print(f"Loading method for dataset '{dataset_name}' is not defined.")
            continue

        horizons = dataset_cfg.horizons
        mean_smape_results = []  # Will store mean sMAPE per horizon

        # Loop over forecast horizons for this dataset
        for horizon in horizons:
            print(f"\n--- Evaluating for horizon: {horizon} ---")
            Y_train_df, Y_test_df = train_test_split(Y_ALL, horizon=horizon)

            # For each active model, run the experiment.
            smape_list = []  # To store sMAPE for each model configuration for this horizon
            for model_name in cfg.active_models:
                print(f"\n>>> Running model: {model_name}")
                model_config = cfg.models.get(model_name)
                if model_config is None:
                    print(f"Model configuration '{model_name}' not found in cfg.models!")
                    continue

                # Retrieve model hyper-parameters (in case some are lists for testing multiple configurations)
                params = model_config.params

                # Determine the number of configurations for this model (based on one of the parameters)
                if isinstance(params.loss, list):
                    n_configs = len(params.loss)
                else:
                    n_configs = 1

                for i in range(n_configs):
                    # Extract each hyper-parameter, using the helper function to handle lists.
                    h_val = horizon
                    input_size_val = get_config_value(params.input_size, i)
                    dropout_val = get_config_value(params.dropout, i)
                    loss_str = get_config_value(params.loss, i)
                    valid_loss_str = get_config_value(params.valid_loss, i)
                    early_stop = get_config_value(params.early_stop_patience_steps, i)
                    batch_size_val = get_config_value(params.batch_size, i)

                    # Dynamically select and initialize the model.
                    # For example, if model_name is "simpleMoe", we call the corresponding SimpleMoe class.
                    if model_name.lower() == "simplemoe":
                        model_instance = SimpleMoe(
                            h=h_val,
                            input_size=input_size_val,
                            dropout=dropout_val,
                            loss=eval(loss_str)(),        # For instance, eval("SMAPE")() creates an instance of SMAPE.
                            valid_loss=eval(valid_loss_str)(),
                            early_stop_patience_steps=early_stop,
                            batch_size=batch_size_val
                        )
                    else:
                        print(f"Model '{model_name}' is not implemented.")
                        continue

                    # Forecasting: instantiate NeuralForecast with the current model.
                    fcst = NeuralForecast(
                        models=[model_instance],
                        freq=cfg.forecast.default_forecast.freq
                    )

                    fcst.fit(
                        df=Y_train_df,
                        static_df=None,
                        val_size=horizon
                    )
                    forecasts = fcst.predict(futr_df=Y_test_df)

                    # Assuming the forecast column is named after the model class, e.g., "SimpleMoe"
                    forecast_col = model_instance.__class__.__name__
                    Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id', 'ds'])

                    # Evaluate sMAPE. We reshape y_true and y_hat assuming one row per series.
                    y_true = Y_test_df['y'].values
                    try:
                        y_hat = Y_hat_df[forecast_col].values
                    except KeyError:
                        print(f"Forecast column '{forecast_col}' not found in predictions!")
                        continue

                    n_series = Y_test_df['unique_id'].nunique()
                    try:
                        y_true = y_true.reshape(n_series, -1)
                        y_hat = y_hat.reshape(n_series, -1)
                    except Exception as e:
                        print("Error reshaping arrays:", e)
                        continue

                    current_smape = smape(y_true, y_hat)
                    print(f"Model '{model_name}' config {i}: sMAPE = {current_smape:.3f}")
                    smape_list.append(current_smape)

            # Compute the mean sMAPE for this horizon across all model configurations.
            if smape_list:
                mean_smape = np.mean(smape_list)
                mean_smape_results.append(mean_smape)
                print(f"Mean sMAPE for horizon {horizon}: {mean_smape:.3f}")
            else:
                mean_smape_results.append(np.nan)
                print(f"No valid model runs for horizon {horizon}.")

        # ---- Plotting the mean sMAPE vs. horizon for the current dataset ----
        plt.figure(figsize=(10, 6))
        plt.plot(horizons, mean_smape_results, marker='o', linestyle='-')
        plt.xlabel("Forecast Horizon", fontsize=14)
        plt.ylabel("Mean sMAPE", fontsize=14)
        plt.title(f"Mean sMAPE vs. Forecast Horizon ({dataset_name})", fontsize=16)
        plt.grid(True)
        plt.xticks(horizons)
        plt.savefig(f"mean_smape_vs_horizon_{dataset_name}.png")
        plt.show()

        # ---- (Optional) Plot forecasts for the last experiment run ----
        # Here we simply plot the forecasts from the last run (for illustration).
        fig, ax = plt.subplots(1, 1, figsize=cfg.plot.default_plot.figsize)
        # We assume the variable 'forecasts' and 'Y_train_df' are from the last iteration.
        forecast_col = model_instance.__class__.__name__
        Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id', 'ds'])
        # Combine train and test for plotting.
        plot_df = pd.concat([Y_train_df, pd.concat([Y_test_df, Y_hat_df], axis=1)], axis=0)
        
        # Plot a few series (e.g., last 10 unique ids).
        for unique_id in plot_df['unique_id'].unique()[-10:]:
            temp_df = plot_df[plot_df['unique_id'] == unique_id]
            plt.plot(temp_df['ds'], temp_df['y'], label=f"True {unique_id}")
            plt.plot(temp_df['ds'], temp_df[forecast_col], label=f"Forecast {unique_id}")
        
        ax.set_title(cfg.plot.default_plot.title, fontsize=22)
        ax.set_ylabel(cfg.plot.default_plot.ylabel, fontsize=20)
        ax.set_xlabel(cfg.plot.default_plot.xlabel, fontsize=20)
        ax.legend(prop={'size': cfg.plot.default_plot.legend_font_size})
        ax.grid(cfg.plot.default_plot.grid)
        fig.savefig(f"{dataset_name}_{cfg.plot.default_plot.save_path}")
        plt.show()


if __name__ == "__main__":
    main()
