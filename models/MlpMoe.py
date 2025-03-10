import torch
import torch.nn as nn

from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.losses.pytorch import MAE

from .pooling_methods import SparsePooling

# type hints
from typing import Optional


class MLPMoe(BaseWindows):

    # Class attributes
    SAMPLING_TYPE = "windows"
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True

    def __init__(
        self,
        h,
        input_size,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y=False,
        num_layers=2,
        hidden_size=1024,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size=1024,
        inference_windows_batch_size=-1,
        start_padding_enabled=False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs
    ):

        # Inherit BaseWindows class
        super(MLPMoe, self).__init__(
            h=h,
            input_size=input_size,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            drop_last_loader=drop_last_loader,
            random_seed=random_seed,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs
        )

        # Architecture
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        input_size_first_layer = (
            input_size
            + self.hist_exog_size * input_size
            + self.futr_exog_size * (input_size + h)
            + self.stat_exog_size
        )

        # MultiLayer Perceptron
        layers = [
            nn.Linear(in_features=input_size_first_layer, out_features=hidden_size)
        ]
        for i in range(num_layers - 1):
            layers += [nn.Linear(in_features=hidden_size, out_features=hidden_size)]
        self.mlp = nn.ModuleList(layers)

        # Adapter with Loss dependent dimensions
        # self.out = nn.Linear(
        #     in_features=hidden_size, out_features=h * self.loss.outputsize_multiplier
        # )
        self.out = SparsePooling(
            experts=[nn.Linear(hidden_size, h * self.loss.outputsize_multiplier, bias=True) for _ in range(8)],
            gate=nn.Linear(in_features=hidden_size, out_features=8),
            out_features=h * self.loss.outputsize_multiplier,
            k=1,
            unpack=False,
        )

    def forward(self, windows_batch):

        # Parse windows_batch
        insample_y = windows_batch["insample_y"]
        futr_exog = windows_batch["futr_exog"]
        hist_exog = windows_batch["hist_exog"]
        stat_exog = windows_batch["stat_exog"]

        # Flatten MLP inputs [B, L+H, C] -> [B, (L+H)*C]
        # Contatenate [ Y_t, | X_{t-L},..., X_{t} | F_{t-L},..., F_{t+H} | S ]
        batch_size = len(insample_y)
        if self.hist_exog_size > 0:
            insample_y = torch.cat(
                (insample_y, hist_exog.reshape(batch_size, -1)), dim=1
            )

        if self.futr_exog_size > 0:
            insample_y = torch.cat(
                (insample_y, futr_exog.reshape(batch_size, -1)), dim=1
            )

        if self.stat_exog_size > 0:
            insample_y = torch.cat(
                (insample_y, stat_exog.reshape(batch_size, -1)), dim=1
            )

        y_pred = insample_y.clone()
        for layer in self.mlp:
            y_pred = torch.relu(layer(y_pred))
        y_pred = self.out(y_pred)

        y_pred = y_pred.reshape(batch_size, self.h, self.loss.outputsize_multiplier)
        y_pred = self.loss.domain_map(y_pred)
        return y_pred
