import pandas as pd
from datasetsforecast.m3 import M3  
from datasetsforecast.m4 import M4  
import hydra
from omegaconf import DictConfig

from neuralforecast.tsdataset import TimeSeriesDataset

#### auto models

from models.auto.AutoNbeatsMoe import AutoNBEATSMoE


def load_dataset(dataset_name: str, dataset_cfg: DictConfig):
    """Load dataset based on dataset_name and its configuration."""
    if dataset_name == "m3":
        print("Loading m3_monthly dataset...")
        return M3.load(
            directory=dataset_cfg.directory,
            group=dataset_cfg.group)[0]
    elif dataset_name == "m4":
        print("Loading m4_monthly dataset...")
        df = M4.load(
            directory=dataset_cfg.directory,
            group=dataset_cfg.group)[0]
        
        # Convert the 'ds' to integer
        df['ds'] = pd.to_datetime(df['ds']).astype(int)

        return df
    else:
        raise ValueError(
            f"Loading method for dataset '{dataset_name}' is not defined.")

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

def get_model(name: str, study_name: str):
    if name.lower() == "nbeatsmoe":
        return AutoNBEATSMoE(
            h=18, 
            num_samples=20,
            backend="optuna",
            optuna_kargs={
            "study_name": study_name,
            "storage": "sqlite:///c:/Users/ricar/mixture_of_experts_time_series/db/nbeatsmoe.db",
            "load_if_exists": True
            }
        )
    else:
        raise ValueError(
            f"Model '{name}' is not defined.")


@hydra.main(config_path="conf", config_name="hyper.yaml")
def main(cfg: DictConfig):
    Y_ALL = load_dataset(cfg.dataset.name, cfg.dataset)

    Y_train_df, Y_test_df = train_test_split(Y_ALL, cfg.horizon)

    dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)

    study_name = f"{cfg.model.name}_{cfg.dataset.name}_{cfg.horizon}"
    model = get_model(cfg.model.name, study_name)

    model.fit(dataset)



if __name__ == "__main__":
    main()



## TODO:
## 1. fazer o yaml
## 3. verificar se está a guardar os resultados dos hyperparametros no sql
## 2. chamar o modelo associado no yaml com dataset associado
## 4. fazer o script de teste
## 5. guardar os resultados num ficheiro csv