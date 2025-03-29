import hydra
from omegaconf import DictConfig
from utils import load_dataset, train_test_split
from typing import Dict

from neuralforecast.tsdataset import TimeSeriesDataset

from omegaconf import ListConfig

from ray import tune
from neuralforecast.auto import BaseAuto

#### auto models

from models.auto.AutoNbeatsMoe import AutoNBEATSMoE
from models.auto.AutoInformerMoe import AutoInformerMoe
from models.auto.AutoMlpMoe import AutoMLPMoe
from neuralforecast.auto import AutoNBEATS, AutoVanillaTransformer, AutoMLP


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
            # "stack_types": tune.choice([["identity", "trend", "seasonality"], ["identity", "trend"]]),
            "mlp_units": tune.choice([3 * [[pow(2, 2+x), pow(2, 2+x)]] for x in range(9)]),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "scaler_type": tune.choice(["identity", "minmax", "robust", "standard"]),
            "max_steps": tune.choice([1000, 2500, 5000]),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "windows_batch_size": tune.choice([128, 256, 512, 1024]),
            "random_seed": tune.randint(1, 20),
            "early_stop_patience_steps": tune.choice([5, 10, 20]),
            "start_padding_enabled": tune.choice([True]),
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
    elif name.lower() == "informermoe":
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
    elif name.lower() == "mlp":
        return AutoMLP(
            h=horizon, 
            num_samples=20,
            backend="optuna",
            optuna_kargs={
            "study_name": study_name,
            "storage": "sqlite:///c:/Users/ricar/mixture_of_experts_time_series/db/study.db",
            "load_if_exists": True
            }
        )
    elif name.lower() == "mlpmoe":
        return AutoMLPMoe(
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


def run_pipeline(dataset_info: Dict[str, str], model_info: Dict[str, str], horizon: int):
    Y_ALL = load_dataset(dataset_info["name"], dataset_info)

    Y_train_df, Y_test_df = train_test_split(Y_ALL, horizon)

    dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)

    study_name = f"{model_info['name']}_{dataset_info['name']}_{dataset_info['group']}_{horizon}"
    model = get_model(model_info["name"], horizon, study_name)
    
    model.fit(dataset)


def _get_all_groups_and_freq_of_dataset(dataset_info: Dict[str, str]):
    if dataset_info["name"] == "m3":
        return [
            {"name": dataset_info["name"], "directory": dataset_info["directory"], "group": "Monthly", "freq": "M"},
            {"name": dataset_info["name"], "directory": dataset_info["directory"], "group": "Quarterly", "freq": "Q"},
            {"name": dataset_info["name"], "directory": dataset_info["directory"], "group": "Yearly", "freq": "Y"},
            {"name": dataset_info["name"], "directory": dataset_info["directory"], "group": "Other", "freq": "O"},
        ]
    else:
        raise ValueError(f"Dataset '{dataset_info['name']}' is not defined.")

@hydra.main(config_path="conf", config_name="hyper.yaml")
def main(cfg: DictConfig):
    
    dataset_info = cfg.dataset
    model_info = cfg.model
    horizon = cfg.horizon

    models_names = model_info["name"] if isinstance(model_info["name"], ListConfig) else [model_info["name"]]
    horizons = horizon if isinstance(horizon, ListConfig) else [horizon]

    if cfg.get("all", True):
        ## get all the possible groups and frequencies of the dataset
        groups_and_freq = _get_all_groups_and_freq_of_dataset(dataset_info)
        for group_and_freq in groups_and_freq:
            for model_name in models_names:
                for h in horizons:
                    run_pipeline(group_and_freq, {"name": model_name}, h)
    else:
        for model_name in models_names:
            for h in horizons:
                run_pipeline(dataset_info, {"name": model_name}, h)

if __name__ == "__main__":
    main()