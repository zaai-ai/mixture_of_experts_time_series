import pandas as pd
import hydra
from omegaconf import DictConfig
import random
from copy import deepcopy

from typing import Any

from neuralforecast.common._base_windows import BaseModel
from neuralforecast.tsdataset import TimeSeriesDataset

import optuna
from neuralforecast.losses.numpy import smape, mae, mse

from utils import load_dataset, train_test_split

from models.NBeatsMoe import NBeatsMoe
from models.InformerMoe import InformerMoe
from neuralforecast.models import VanillaTransformer
from neuralforecast.models import NBEATS

def get_instance(name: str, best_params: dict[str, Any], horizon: int) -> BaseModel:
    if name.lower() == "nbeatsmoe":
        return NBeatsMoe(h=horizon, **best_params)
    elif name.lower() == "nbeats":
        return NBEATS(h=horizon, **best_params)
    elif name.lower() == "informermoe":
        return InformerMoe(h=horizon, **best_params)
    elif name.lower() == "vanillatransformer":
        return VanillaTransformer(h=horizon, **best_params)
    else:
        raise ValueError(
            f"Model '{name}' is not defined.")

@hydra.main(config_path="conf", config_name="hyper.yaml")
def main(cfg: DictConfig):
    Y_ALL = load_dataset(cfg.dataset.name, cfg.dataset)

    Y_train_df, Y_test_df = train_test_split(Y_ALL, cfg.horizon)

    dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)
    test_dataset, *_ = TimeSeriesDataset.from_df(Y_test_df)

    study_name = f"{cfg.model.name}_{cfg.dataset.name}_{cfg.horizon}"
    
    study = optuna.load_study(
        study_name=study_name,
        storage="sqlite:///c:/Users/ricar/mixture_of_experts_time_series/db/study.db",
    )

    results = {"smape": [], "mae": [], "mse": []}

    for _ in range(10):
        random_seed = random.randint(1, 1000)
        best_params = deepcopy(study.best_params)
        best_params["random_seed"] = random_seed

        model = get_instance(cfg.model.name, best_params, cfg.horizon)
        model.fit(dataset)

        # Evaluate on the test dataset
        Y_pred_df = model.predict(test_dataset)
        smape_e = smape(Y_test_df['y'].values, Y_pred_df)
        mae_e = mae(Y_test_df['y'].values, Y_pred_df)
        mse_e = mse(Y_test_df['y'].values, Y_pred_df)

        # TODO: FIX THIS
        # results["smape"].append(smape_e)
        # results["mae"].append(mae_e)
        # results["mse"].append(mse_e)

        print(f"results: {results}")

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Calculate standard deviation and median for each metric
    std_dev = results_df.std()
    median = results_df.median()

    print("Standard Deviation:")
    print(std_dev)
    print("\nMedian:")
    print(median)
        


if __name__ == "__main__":
    main()
