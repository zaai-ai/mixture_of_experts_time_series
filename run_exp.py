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

def get_instance(name: str, best_params: dict[str, Any], horizon: int) -> BaseModel:
    if name.lower() == "nbeatsmoe":
        return NBeatsMoe(h=horizon, **best_params)
    elif name.lower() == "nbeats":
        best_params["scaler_type"] = "identity"
        return NBEATS(h=horizon, **best_params)
    elif name.lower() == "nbeatsstackmoe":
        best_params["scaler_type"] = "identity"
        return NBeatsStackMoe(h=horizon, **best_params)
    elif name.lower() == "autoinformermoe":
        return InformerMoe(h=horizon, **best_params)
    elif name.lower() == "vanillatransformer":
        return VanillaTransformer(h=horizon, **best_params)
    else:
        raise ValueError(
            f"Model '{name}' is not defined.")

def save_results_summary(cfg, std_dev, median, results_file="results_summary.csv"):
    """
    Saves model evaluation metrics to a CSV file.

    Parameters:
        cfg (object): Configuration object containing model and dataset details.
        std_dev (dict): Dictionary containing standard deviation values for metrics.
        median (dict): Dictionary containing median values for metrics.
        results_file (str): Path to the results summary CSV file.
    """
    new_row = {
        "model_name": cfg.model.name,
        "dataset": cfg.dataset.name,
        "group": cfg.dataset.group,
        "horizon": cfg.horizon,
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


@hydra.main(config_path="conf", config_name="hyper.yaml")
def main(cfg: DictConfig):
    Y_ALL = load_dataset(cfg.dataset.name, cfg.dataset)

    Y_train_df, Y_test_df = train_test_split(Y_ALL, cfg.horizon)

    dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)
    test_dataset, *_ = TimeSeriesDataset.from_df(Y_test_df)

    study_name = f"{cfg.model.name}_{cfg.dataset.name}_{cfg.dataset.group}_{cfg.horizon}"
    
    if cfg.model.name.lower() == "nbeatsstackmoe":
        study_name = study_name.replace("stackmoe", "")

    study = optuna.load_study(
        study_name=study_name,
        storage="sqlite:///c:/Users/ricar/mixture_of_experts_time_series/db/study.db",
    )

    results = {"smape": [], "mae": [], "mse": []}

    list_random_seeds = random.sample(range(1, 1000), 20)
    for i in range(20):
        random_seed = list_random_seeds[i]
        best_params = deepcopy(study.best_params)
        best_params["random_seed"] = random_seed

        print(f"best_params: {best_params}")

        model = get_instance(cfg.model.name, best_params, cfg.horizon)

        fcst = NeuralForecast(models=[model], freq=cfg.dataset.freq)
        fcst.fit(df=Y_train_df, static_df=None, val_size=cfg.horizon)

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
    save_results_summary(cfg, std_dev, median, results_file="C:\\Users\\ricar\\mixture_of_experts_time_series\\results_summary.csv")

if __name__ == "__main__":
    main()
