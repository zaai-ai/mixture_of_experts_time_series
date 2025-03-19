import pandas as pd
from datasetsforecast.m3 import M3  
from datasetsforecast.m4 import M4  
import hydra
from omegaconf import DictConfig

from typing import Any

from neuralforecast.common._base_windows import BaseModel
from neuralforecast.tsdataset import TimeSeriesDataset

import optuna

from optuna.visualization import plot_optimization_history

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

    study_name = f"{cfg.model.name}_{cfg.dataset.name}_{cfg.horizon}"
    
    study = optuna.load_study(
        study_name=study_name,
        storage="sqlite:///c:/Users/ricar/mixture_of_experts_time_series/db/study.db",
    )

    print(study.best_params)

    for _ in range(10):
        model = get_instance(cfg.model.name, study.best_params, cfg.horizon)
        # model.fit(dataset)
        


if __name__ == "__main__":
    main()
