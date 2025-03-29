from os import cpu_count
import torch

from ray import tune

from ray.tune.search.basic_variant import BasicVariantGenerator
from neuralforecast.losses.pytorch import MAE

from models.MlpMoe import MLPMoe
from neuralforecast.auto import BaseAuto

class AutoMLPMoe(BaseAuto):

    default_config = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "hidden_size": tune.choice([256, 512, 1024]),
        "num_layers": tune.randint(2, 6),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.choice([500, 1000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
        "nr_experts": tune.choice([pow(2,x) for x in range(1, 4)]),
        "top_k": tune.choice([pow(2,x) for x in range(0, 4)]),
       
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

        super(AutoMLPMoe, self).__init__(
            cls_model=MLPMoe,
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
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)

        return config