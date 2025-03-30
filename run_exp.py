import pandas as pd
import hydra
from omegaconf import DictConfig
import random
from copy import deepcopy

from typing import Any

from neuralforecast.common._base_windows import BaseModel
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast import NeuralForecast

import optuna
from neuralforecast.losses.numpy import smape, mae, mse
from neuralforecast.losses.pytorch import HuberLoss

from utils import load_dataset, train_test_split

from models.NBeatsMoe import NBeatsMoe
from models.InformerMoe import InformerMoe
from models.NBeatsStackMoe import NBeatsStackMoe
from neuralforecast.models import VanillaTransformer
from neuralforecast.models import NBEATS

from omegaconf import ListConfig

def get_instance(name: str, best_params: dict[str, Any], horizon: int) -> BaseModel:
    if name.lower() == "nbeatsmoe":
        return NBeatsMoe(h=horizon, **best_params)
    elif name.lower() == "nbeats":
        best_params["scaler_type"] = "identity"
        return NBEATS(h=horizon, **best_params)
    elif name.lower() == "nbeatsstackmoe":
        best_params["scaler_type"] = "identity"
        return NBeatsStackMoe(h=horizon, **best_params)
    elif name.lower() == "informermoe":
        return InformerMoe(h=horizon, **best_params)
    elif name.lower() == "vanillatransformer":
        return VanillaTransformer(h=horizon, **best_params)
    else:
        raise ValueError(
            f"Model '{name}' is not defined.")

def save_results_summary(model_name, dataset, horizon, std_dev, median, results_file="results_summary.csv"):
    """
    Saves model evaluation metrics to a CSV file.

    Parameters:
        cfg (object): Configuration object containing model and dataset details.
        std_dev (dict): Dictionary containing standard deviation values for metrics.
        median (dict): Dictionary containing median values for metrics.
        results_file (str): Path to the results summary CSV file.
    """
    new_row = {
        "model_name": model_name,
        "dataset": dataset.name,
        "group": dataset.group,
        "horizon": horizon,
        "std_dev_smape": std_dev["smape"],
        "std_dev_mae": std_dev["mae"],
        "std_dev_mse": std_dev["mse"],
        "median_smape": median["smape"],
        "median_mae": median["mae"],
        "median_mse": median["mse"],
    }

    try:
        results_summary = pd.read_csv(results_file)
        results_summary = pd.concat([results_summary, pd.DataFrame([new_row])], ignore_index=True)
    except FileNotFoundError:
        results_summary = pd.DataFrame([new_row])

    results_summary.to_csv(results_file, index=False)


@hydra.main(config_path="conf", config_name="exp.yaml")
def main(cfg: DictConfig):
    Y_ALL = load_dataset(cfg.dataset.name, cfg.dataset)

    if type(Y_ALL) == tuple:
        Y_ALL, cfg.horizon, n_lags, cfg.dataset.group, _ = Y_ALL
        cfg.dataset.freq = cfg.dataset.group
        cfg.dataset.name = cfg.dataset.name.replace(" ", "_")

    # dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)
    # test_dataset, *_ = TimeSeriesDataset.from_df(Y_test_df)

    model_info = cfg.model
    horizon = cfg.horizon

    models_names = model_info["name"] if isinstance(model_info["name"], ListConfig) else [model_info["name"]]
    horizons = horizon if isinstance(horizon, ListConfig) else [horizon]

    for model_name in models_names:
        for horizon in horizons:
            Y_train_df, Y_test_df = train_test_split(Y_ALL, horizon)

            study_name = f"{model_name}_{cfg.dataset.name}_{cfg.dataset.group}_{horizon}"
            
            if model_name.lower() == "nbeatsstackmoe":
                study_name = study_name.replace("stackmoe", "")

            tentatives = 0
            while tentatives < 24:
                try:
                    study = optuna.load_study(
                        study_name=study_name,
                        storage="sqlite:///c:/Users/ricar/mixture_of_experts_time_series/db/study.db",
                    )
                    break
                except KeyError:
                    print(f"Error: There is no study with the name '{study_name}'.")
                    # change study name to search for a new horizon
                    tentatives += 1
                    study_name = f"{model_name}_{cfg.dataset.name}_{cfg.dataset.group}_{horizon + tentatives}"
                    if model_name.lower() == "nbeatsstackmoe":
                        study_name = study_name.replace("stackmoe", "")

            if tentatives == 10:
                print("Error: There is no study available")

            results = {"smape": [], "mae": [], "mse": []}

            list_random_seeds = random.sample(range(1, 1000), 10)
            for i in range(10):
                random_seed = list_random_seeds[i]
                best_params = deepcopy(study.best_params)
                best_params["random_seed"] = random_seed

                print(f"best_params: {best_params}")

                model = get_instance(model_name, best_params, horizon)

                fcst = NeuralForecast(models=[model], freq=cfg.dataset.freq)
                fcst.fit(df=Y_train_df, static_df=None, val_size=horizon)

                # Evaluate on the test dataset
                Y_pred_df = fcst.predict(futr_df=Y_test_df)

                y_true = Y_test_df['y'].values
                y_hat = Y_pred_df[model.__class__.__name__].values

                n_series = Y_test_df['unique_id'].nunique()
                y_true = y_true.reshape(n_series, -1)
                y_hat = y_hat.reshape(n_series, -1)

                smape_e = smape(y_true, y_hat)
                mae_e = mae(y_true, y_hat)
                mse_e = mse(y_true, y_hat)

                results["smape"].append(smape_e)
                results["mae"].append(mae_e)
                results["mse"].append(mse_e)

                print(f"results: {results}")

            # Convert results to a DataFrame
            results_df = pd.DataFrame(results)

            # Calculate standard deviation and median for each metric
            std_dev = results_df.std()
            median = results_df.median()

            print("Standard Deviation:")
            print(std_dev.round(4))
            print("\nMedian:")   
            print(median.round(4))

            # Save results to a CSV file with the specified columns
            save_results_summary(model_name, cfg.dataset, horizon, std_dev, median, results_file="C:\\Users\\ricar\\mixture_of_experts_time_series\\results_summary.csv")

if __name__ == "__main__":
    main()
