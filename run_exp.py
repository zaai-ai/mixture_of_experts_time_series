import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.losses.numpy import smape
from datasetsforecast.m3 import M3
from neuralforecast.models import RMoK
from neuralforecast.losses.pytorch import SMAPE
import hydra
from omegaconf import DictConfig

# Import your model classes (here we assume the module name matches the model name)
from models.SimpleMoe import SimpleMoe


def load_dataset(dataset_name: str, dataset_cfg: DictConfig):
    """Load dataset based on dataset_name and its configuration."""
    if dataset_name == "m3_monthly":
        print("Loading m3_monthly dataset...")
        return M3.load(directory=dataset_cfg.directory, group=dataset_cfg.group)[0]
    else:
        raise ValueError(f"Loading method for dataset '{dataset_name}' is not defined.")


def train_test_split(df: pd.DataFrame, horizon: int):
    """Split the dataframe into training and test sets by horizon."""
    groups = df.groupby('unique_id')
    train_list, test_list = [], []
    for _, group_df in groups:
        group_df = group_df.sort_values('ds')
        train_list.append(group_df.head(-horizon))
        test_list.append(group_df.tail(horizon))
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df, test_df


def get_config_value(cfg_val, idx):
    """
    Helper: If cfg_val is a list then return the idx-th element;
    otherwise return cfg_val.
    """
    if isinstance(cfg_val, list):
        return cfg_val[idx % len(cfg_val)]
    return cfg_val


def run_model_experiment(
    model_name: str,
    model_config: DictConfig,
    Y_train_df: pd.DataFrame,
    Y_test_df: pd.DataFrame,
    horizon: int,
    freq: str,
    config_idx: int = 0
):
    """
    For a given model configuration, initialize the model, run forecast, and compute sMAPE.
    Returns the computed sMAPE, the forecasts, and the model instance.
    """
    params = model_config.params

    # Extract parameters (for those that might be lists, we use the helper function)
    # Note: For this example, we assume that 'h' is set to the forecast horizon.
    h_val = horizon  # here you might decide to use horizon as a model parameter
    input_size_val = get_config_value(params.input_size, config_idx)
    dropout_val = get_config_value(params.dropout, config_idx)
    loss_str = get_config_value(params.loss, config_idx)
    valid_loss_str = get_config_value(params.valid_loss, config_idx)
    early_stop = get_config_value(params.early_stop_patience_steps, config_idx)
    batch_size_val = get_config_value(params.batch_size, config_idx)

    # Initialize model instance based on model_name.
    if model_name.lower() == "simplemoe":
        model_instance = SimpleMoe(
            h=h_val,
            input_size=input_size_val,
            dropout=dropout_val,
            loss=eval(loss_str)(),         # e.g., eval("SMAPE")() creates an instance of SMAPE.
            valid_loss=eval(valid_loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val
        )        
    else:
        raise NotImplementedError(f"Model '{model_name}' is not implemented.")

    # Instantiate NeuralForecast and run forecast.
    fcst = NeuralForecast(models=[model_instance], freq=freq)
    fcst.fit(df=Y_train_df, static_df=None, val_size=horizon)
    forecasts = fcst.predict(futr_df=Y_test_df)

    # Extract forecast column (assumes column is named after model class)
    forecast_col = model_instance.__class__.__name__
    Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id', 'ds'])

    # Evaluate sMAPE.
    y_true = Y_test_df['y'].values
    try:
        y_hat = Y_hat_df[forecast_col].values
    except KeyError:
        raise KeyError(f"Forecast column '{forecast_col}' not found in predictions!")

    n_series = Y_test_df['unique_id'].nunique()
    try:
        y_true = y_true.reshape(n_series, -1)
        y_hat = y_hat.reshape(n_series, -1)
    except Exception as e:
        raise ValueError("Error reshaping arrays") from e

    current_smape = smape(y_true, y_hat)
    return current_smape, forecasts, model_instance


def plot_mean_smape(horizons, mean_smape_results, dataset_name: str):
    """Plot the mean sMAPE versus forecast horizons."""
    plt.figure(figsize=(10, 6))
    plt.plot(horizons, mean_smape_results, marker='o', linestyle='-')
    plt.xlabel("Forecast Horizon", fontsize=14)
    plt.ylabel("Mean sMAPE", fontsize=14)
    plt.title(f"Mean sMAPE vs. Forecast Horizon ({dataset_name})", fontsize=16)
    plt.grid(True)
    plt.xticks(horizons)
    plt.savefig(f"mean_smape_vs_horizon_{dataset_name}.png")
    plt.show()


def plot_forecasts(Y_train_df, Y_test_df, forecasts, model_instance, dataset_name: str, plot_cfg: DictConfig):
    """
    Plot the ground truth and the forecasts side by side.
    Left subplot: full ground truth (training + test)
    Right subplot: forecasts on the test period.
    """
    forecast_col = model_instance.__class__.__name__
    # Ensure forecasts has the 'unique_id', 'ds', and forecast_col columns
    forecasts = forecasts.reset_index(drop=False)
    
    # Combine training and test data for the ground truth
    gt_df = pd.concat([Y_train_df, Y_test_df], axis=0)
    # gt_df = Y_test_df.copy()
    
    # Create a figure with 2 subplots side by side.
    fig, axs = plt.subplots(1, 2, figsize=(plot_cfg.figsize[0]*2, plot_cfg.figsize[1]))

    unique_ids = gt_df['unique_id'].unique()[-10:]  # Plot only the last 10 series
    
    # --- Left subplot: Ground Truth ---
    for unique_id in unique_ids:
        temp_df = gt_df[gt_df['unique_id'] == unique_id]
        axs[0].plot(temp_df['ds'], temp_df['y'], label=f"Series {unique_id}")
    axs[0].set_title("Ground Truth")
    axs[0].set_xlabel(plot_cfg.xlabel)
    axs[0].set_ylabel(plot_cfg.ylabel)
    axs[0].grid(plot_cfg.grid)
    axs[0].legend(fontsize=plot_cfg.legend_font_size)
    
    # --- Right subplot: Forecasts ---
    # We assume forecasts are only available for the test period.
    for unique_id in unique_ids:
        temp_df = forecasts[forecasts['unique_id'] == unique_id]
        axs[1].plot(temp_df['ds'], temp_df[forecast_col], label=f"Series {unique_id}")
    axs[1].set_title("Forecasts")
    axs[1].set_xlabel(plot_cfg.xlabel)
    axs[1].set_ylabel(plot_cfg.ylabel)
    axs[1].grid(plot_cfg.grid)
    axs[1].legend(fontsize=plot_cfg.legend_font_size)
    
    # Set an overall title, adjust layout, and save the figure.
    fig.suptitle(plot_cfg.title, fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{dataset_name}_{plot_cfg.save_path}")
    plt.show()



@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):

    # Loop over each active dataset specified in the config.
    for dataset_name in cfg.active_datasets:
        print(f"\n==== Running experiment for dataset: {dataset_name} ====")
        dataset_cfg = cfg.data.get(dataset_name)
        if dataset_cfg is None:
            print(f"Dataset configuration '{dataset_name}' not found in config.data!")
            continue

        try:
            Y_ALL = load_dataset(dataset_name, dataset_cfg)
        except Exception as e:
            print(e)
            continue

        horizons = dataset_cfg.horizons
        mean_smape_results = []  # to store mean sMAPE per horizon

        # Loop over each forecast horizon.
        for horizon in horizons:
            print(f"\n--- Evaluating for horizon: {horizon} ---")
            Y_train_df, Y_test_df = train_test_split(Y_ALL, horizon=horizon)
            smape_list = []

            # Loop over each active model.
            for model_name in cfg.active_models:
                print(f"\n>>> Running model: {model_name}")
                model_config = cfg.models.get(model_name)
                if model_config is None:
                    print(f"Model configuration '{model_name}' not found in cfg.models!")
                    continue

                # Determine number of configurations to run for this model.
                # (For fixed values, this will be 1.)
                if isinstance(horizons, list):
                    n_configs = len(horizons)
                else:
                    n_configs = 1

                for i in range(n_configs):
                    try:
                        current_smape, forecasts, model_instance = run_model_experiment(
                            model_name,
                            model_config,
                            Y_train_df,
                            Y_test_df,
                            horizon,
                            cfg.forecast.default_forecast.freq,
                            config_idx=i
                        )
                        print(f"Model '{model_name}' config {i}: sMAPE = {current_smape:.3f}")
                        smape_list.append(current_smape)
                    except Exception as e:
                        print(f"Error running model '{model_name}' config {i}: {e}")
                        continue

            # Compute mean sMAPE for this horizon.
            if smape_list:
                mean_smape = np.mean(smape_list)
                mean_smape_results.append(mean_smape)
                print(f"Mean sMAPE for horizon {horizon}: {mean_smape:.3f}")
            else:
                mean_smape_results.append(np.nan)
                print(f"No valid model runs for horizon {horizon}.")

        # Plot mean sMAPE vs. horizons.
        plot_mean_smape(horizons, mean_smape_results, dataset_name)

        # Optionally, plot the forecasts from the last experiment run.
        try:
            plot_forecasts(Y_train_df, Y_test_df, forecasts, model_instance, dataset_name, cfg.plot.default_plot)
        except Exception as e:
            print(f"Error plotting forecasts: {e}")


if __name__ == "__main__":
    main()
