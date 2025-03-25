from os import cpu_count
import torch

from neuralforecast.auto import AutoNBEATS
from ray import tune

from models.NBeatsMoe import NBeatsMoe

from ray.tune.search.basic_variant import BasicVariantGenerator
from neuralforecast.losses.pytorch import MAE
from neuralforecast.auto import BaseAuto



class AutoNBEATSMoE(BaseAuto):

    default_config = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        # "stack_types": tune.choice([["identity", "trend", "seasonality"], ["identity", "trend"]]),
        "mlp_units": tune.choice([3 * [[pow(2, 2+x), pow(2, 2+x)]] for x in range(8)]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice(["identity"]),
        "max_steps": tune.choice([1000, 2500, 5000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "random_seed": tune.randint(1, 20),
        "nr_experts": tune.choice([pow(2,x) for x in range(1, 4)]),
        "top_k": tune.choice([pow(2,x) for x in range(0, 4)]),
        "early_stop_patience_steps": tune.choice([5, 10, 20]),
        "start_padding_enabled": tune.choice([True]),

    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
        optuna_kargs=None,
    ):

        # Define search space, input/output sizes
        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoNBEATSMoE, self).__init__(
            cls_model=NBeatsMoe,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
            optuna_kargs=optuna_kargs,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):

        # def config_with_correct_top_k(trial):
        #     conf = config(trial)
        #     nr_experts = trial.suggest_categorical("nr_experts", [2**x for x in range(1, 5)])
        #     conf["nr_experts"] = nr_experts
        #     conf["top_k"] = trial.suggest_int("top_k", 1, nr_experts)
        #     return conf

        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)


        return config