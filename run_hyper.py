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
from models.auto.AutoNbeatsStackMoe import AutoNBeatsStackMoe
from neuralforecast.auto import AutoNBEATS, AutoVanillaTransformer, AutoMLP

STORAGE = "sqlite:///c:/Users/ricar/mixture_of_experts_time_series/db/study_nbeats_blcs.db"

def get_model(name: str, horizon: int, study_name: str, n_lags: int = None):
    if name.lower() == "nbeatsmoe":
        config = None
        if n_lags:
            config = {
                "h": None,
                "input_size": tune.choice([n_lags]),
                "mlp_units": tune.choice([3 * [[pow(2, 2+x), pow(2, 2+x)]] for x in range(8)]),
                # "learning_rate": tune.loguniform(1e-4, 1e-1),
                "n_blocks": tune.choice([3 * [x] for x in [1, 3, 6, 9]]),
                "scaler_type": tune.choice(["identity"]),
                "shared_weights": tune.choice([True]),
                "max_steps": tune.choice([1000, 2500, 5000, 10000]),
                "batch_size": tune.choice([32, 64, 128, 256]),
                "windows_batch_size": tune.choice([128, 256, 512, 1024]),
                "random_seed": tune.randint(1, 20),
                "nr_experts": tune.choice([pow(2,x) for x in range(1, 4)]),
                "top_k": tune.choice([pow(2,x) for x in range(0, 4)]),
                "early_stop_patience_steps": tune.choice([10, 20]),
                "start_padding_enabled": tune.choice([True]),
            }
            config = BaseAuto._ray_config_to_optuna(config)

        return AutoNBEATSMoE(
            h=horizon, 
            num_samples=20,
            config=config,
            backend="optuna",
            optuna_kargs={
            "study_name": study_name,
            "storage": STORAGE,
            "load_if_exists": True
            }
        )
    if name.lower() == "nbeatsmoeshared":
        config = None
        if n_lags:
            config = {
                "h": None,
                "input_size": tune.choice([n_lags]),
                "mlp_units": tune.choice([3 * [[pow(2, 2+x), pow(2, 2+x)]] for x in range(8)]),
                # "learning_rate": tune.loguniform(1e-4, 1e-1),
                "n_blocks": tune.choice([3 * [x] for x in [1, 3, 6, 9]]),
                "share_experts": tune.choice([True]),
                "shared_weights": tune.choice([True]),
                "scaler_type": tune.choice(["identity"]),       
                "max_steps": tune.choice([1000, 2500, 5000, 10000]),
                "batch_size": tune.choice([32, 64, 128, 256]),
                "windows_batch_size": tune.choice([128, 256, 512, 1024]),
                "random_seed": tune.randint(1, 20),
                "nr_experts": tune.choice([pow(2,x) for x in range(1, 4)]),
                "top_k": tune.choice([pow(2,x) for x in range(0, 4)]),
                "early_stop_patience_steps": tune.choice([10, 20]),
                "start_padding_enabled": tune.choice([True]),
            }
            config = BaseAuto._ray_config_to_optuna(config)

        return AutoNBEATSMoE(
            h=horizon, 
            num_samples=20,
            config=config,
            backend="optuna",
            shared_expert=True,
            optuna_kargs={
            "study_name": study_name,
            "storage": STORAGE,
            "load_if_exists": True
            }
        )
    elif name.lower() == "nbeatsstackmoe":
        config = {
            "input_size": tune.choice([n_lags]) if n_lags else tune.choice([horizon * x for x in [1, 2, 3, 4, 5]]),
            # "stack_types": tune.choice([["identity", "trend", "seasonality"], ["identity", "trend"]]),
            "mlp_units": tune.choice([3 * [[pow(2, 2+x), pow(2, 2+x)]] for x in range(9)]),
            # "learning_rate": tune.loguniform(1e-4, 1e-1),
            "scaler_type": tune.choice(["identity"]),
            "max_steps": tune.choice([1000, 2500, 5000, 10000]),
            "shared_weights": tune.choice([True]),
            "n_blocks": tune.choice([3 * [x] for x in [1, 3, 6, 9]]),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "windows_batch_size": tune.choice([128, 256, 512, 1024]),
            "random_seed": tune.randint(1, 20),
            "early_stop_patience_steps": tune.choice([10, 20]),
            "start_padding_enabled": tune.choice([True]),
        }

        return AutoNBeatsStackMoe(
            h=horizon, 
            config=BaseAuto._ray_config_to_optuna(config),
            num_samples=20,
            backend="optuna",
            optuna_kargs={
            "study_name": study_name,
            "storage": STORAGE,
            "load_if_exists": True
            }
        )
    elif name.lower() == "nbeats":
        config = {
            "input_size": tune.choice([n_lags]) if n_lags else tune.choice([horizon * x for x in [1, 2, 3, 4, 5]]),
            # "stack_types": tune.choice([["identity", "trend", "seasonality"], ["identity", "trend"]]),
            "mlp_units": tune.choice([3 * [[pow(2, 2+x), pow(2, 2+x)]] for x in range(9)]),
            # "learning_rate": tune.loguniform(1e-4, 1e-1),
            "n_blocks": tune.choice([3 * [x] for x in [1, 3, 6, 9]]),
            "shared_weights": tune.choice([True]),
            "scaler_type": tune.choice(["identity"]),
            "max_steps": tune.choice([1000, 2500, 5000, 10000]),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "windows_batch_size": tune.choice([128, 256, 512, 1024]),
            "random_seed": tune.randint(1, 20),
            "early_stop_patience_steps": tune.choice([10, 20]),
            "start_padding_enabled": tune.choice([True]),
        }

        return AutoNBEATS(
            h=horizon, 
            config=BaseAuto._ray_config_to_optuna(config),
            num_samples=20,
            backend="optuna",
            optuna_kargs={
            "study_name": study_name,
            "storage": STORAGE,
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
            "storage": STORAGE,
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
            "storage": STORAGE,
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
            "storage": STORAGE,
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
            "storage": STORAGE,
            "load_if_exists": True
            }
        )
    else:
        raise ValueError(
            f"Model '{name}' is not defined.")


def run_pipeline(dataset_info: Dict[str, str], model_info: Dict[str, str], horizon: int):
    Y_ALL = load_dataset(dataset_info["name"], dataset_info)
    n_lags = None

    if type(Y_ALL) == tuple:
        Y_ALL, horizon, n_lags, dataset_info['group'], _ = Y_ALL

    Y_train_df, Y_test_df = train_test_split(Y_ALL, horizon)

    dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)

    study_name = f"{model_info['name']}_{dataset_info['name']}_{dataset_info['group']}_{horizon}"
    model = get_model(model_info["name"], horizon, study_name, n_lags=n_lags)
    
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