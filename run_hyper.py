import hydra
from omegaconf import DictConfig
from utils import load_dataset, train_test_split

from neuralforecast.tsdataset import TimeSeriesDataset

from ray import tune
from neuralforecast.auto import BaseAuto

#### auto models

from models.auto.AutoNbeatsMoe import AutoNBEATSMoE
from models.auto.AutoInformerMoe import AutoInformerMoe
from neuralforecast.auto import AutoNBEATS, AutoVanillaTransformer


def get_model(name: str, horizon: int, study_name: str):
    if name.lower() == "nbeatsmoe":
        return AutoNBEATSMoE(
            h=horizon, 
            num_samples=20,
            backend="optuna",
            optuna_kargs={
            "study_name": study_name,
            "storage": "sqlite:///c:/Users/ricar/mixture_of_experts_time_series/db/study.db",
            "load_if_exists": True
            }
        )
    elif name.lower() == "nbeats":
        config = {
            "input_size": tune.choice([horizon * x for x in [1, 2, 3, 4, 5]]),
            "stack_types": tune.choice([["identity", "trend", "seasonality"], ["identity", "trend"]]),
            "mlp_units": tune.choice([3 * [[pow(2, 2+x), pow(2, 2+x)]] for x in range(9)]),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "scaler_type": tune.choice([None, "minmax", "robust", "standard"]),
            "max_steps": tune.choice([500, 1000, 5000]),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "windows_batch_size": tune.choice([128, 256, 512, 1024]),
            "random_seed": tune.randint(1, 20),
        }

        return AutoNBEATS(
            h=horizon, 
            config=BaseAuto._ray_config_to_optuna(config),
            num_samples=20,
            backend="optuna",
            optuna_kargs={
            "study_name": study_name,
            "storage": "sqlite:///c:/Users/ricar/mixture_of_experts_time_series/db/study.db",
            "load_if_exists": True
            }
        )
    elif name.lower() == "autoinformermoe":
        return AutoInformerMoe(
            h=horizon, 
            num_samples=20,
            backend="optuna",
            optuna_kargs={
            "study_name": study_name,
            "storage": "sqlite:///c:/Users/ricar/mixture_of_experts_time_series/db/study.db",
            "load_if_exists": True
            }
        )
    elif name.lower() == "vanillatransformer":
        return AutoVanillaTransformer(
            h=horizon, 
            num_samples=20,
            backend="optuna",
            optuna_kargs={
            "study_name": study_name,
            "storage": "sqlite:///c:/Users/ricar/mixture_of_experts_time_series/db/study.db",
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
    model = get_model(cfg.model.name, cfg.horizon, study_name)

    model.fit(dataset)



if __name__ == "__main__":
    main()



## TODO:
## 1. fazer o yaml
## 3. verificar se está a guardar os resultados dos hyperparametros no sql
## 2. chamar o modelo associado no yaml com dataset associado
## 4. fazer o script de teste
## 5. guardar os resultados num ficheiro csv