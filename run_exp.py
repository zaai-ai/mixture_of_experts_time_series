import pandas as pd
from datasetsforecast.m3 import M3  
from datasetsforecast.m4 import M4  
import hydra
from omegaconf import DictConfig

from neuralforecast.tsdataset import TimeSeriesDataset

import optuna

from optuna.visualization import plot_optimization_history

from utils import load_dataset, train_test_split

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



if __name__ == "__main__":
    main()
