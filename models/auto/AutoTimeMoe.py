from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from neuralforecast.auto import BaseAuto
from neuralforecast.losses.pytorch import MAE
from os import cpu_count

from models.TimeMoeAdapted import TimeMoeAdapted, TimeMoeConfig
import torch

class AutoTimeMoe(BaseAuto):

    default_config = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "hidden_size": tune.choice([64, 128, 256, 512]),
        "n_head": tune.choice([2, 4, 8]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        # "max_steps": tune.choice([500, 1000, 2000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "intermediate_size": tune.choice([100, 1000, 5000, 10000, 20000]),
        "random_seed": tune.randint(1, 20),
        "num_experts_per_tok": tune.choice([1, 2, 4]),
        "num_experts": tune.choice([4, 8, 16]),
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
    ):

        # Define search space, input/output sizes
        if config is None:
            config = self.get_default_config(h=h, backend=backend)


        super(AutoTimeMoe, self).__init__(
            cls_model=TimeMoeAdapted,
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
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]

        # config['config'] = TimeMoeConfig(
        #     input_size=config["input_size"],
        #     hidden_size=config["hidden_size"],
        #     intermediate_size=config["intermediate_size"],
        #     num_attention_heads=config["n_head"],
        #     num_experts_per_tok=config["num_experts_per_tok"],
        #     num_experts=config["num_experts"],
        # )
        # del config["n_head"]
        # del config["intermediate_size"]
        # del config["num_experts_per_tok"]
        # del config["num_experts"]

        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)

        return config