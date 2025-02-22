

import torch
import torch.nn as nn

from neuralforecast.losses.pytorch import SMAPE
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.common._modules import RevIN, MLP
from neuralforecast.losses.pytorch import BasePointLoss

from .pooling_methods.methods import *

from neuralforecast.models.nbeats import NBEATS
from typing import Union


# %% ../../nbs/losses.pytorch.ipynb 6
def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    return torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)

# %% ../../nbs/losses.pytorch.ipynb 7
def _weighted_mean(losses, weights):
    """
    Compute weighted mean of losses per datapoint.
    """
    return _divide_no_nan(torch.sum(losses * weights), torch.sum(weights))


class AuxLoss(BasePointLoss):

    current_entropy_loss = 0.0

    def __init__(self, delta: float = 1.0, horizon_weight=None, lambda_entropy=100):
        super(AuxLoss, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )
        self.delta = delta
        self.lambda_entropy = lambda_entropy

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        losses = F.huber_loss(y, y_hat, reduction="none", delta=self.delta)
        balancing_loss = self.lambda_entropy * AuxLoss.current_entropy_loss

        losses = losses + balancing_loss

        # print(balancing_loss, AuxLoss.current_entropy_loss, self.lambda_entropy)

        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)


class SimpleMoe(BaseWindows):
    """
    Simple Mixture of Experts (MoE) model for time series forecasting.
    Attributes:
        SAMPLING_TYPE (str): Type of sampling used, default is 'univariate'.
        EXOGENOUS_FUTR (bool): Indicates if future exogenous variables are used, default is False.
        EXOGENOUS_HIST (bool): Indicates if historical exogenous variables are used, default is False.
        EXOGENOUS_STAT (bool): Indicates if static exogenous variables are used, default is False.
    Args:
        h (int): Forecast horizon.
        input_size (int): Size of the input features.
        dropout (float, optional): Dropout rate, default is 0.1.
        futr_exog_list (list, optional): List of future exogenous variables, default is None.
        hist_exog_list (list, optional): List of historical exogenous variables, default is None.
        stat_exog_list (list, optional): List of static exogenous variables, default is None.
        loss (callable, optional): Loss function, default is SMAPE().
        valid_loss (callable, optional): Validation loss function, default is None.
        max_steps (int, optional): Maximum number of training steps, default is 1000.
        learning_rate (float, optional): Learning rate, default is 1e-3.
        num_lr_decays (int, optional): Number of learning rate decays, default is -1.
        early_stop_patience_steps (int, optional): Early stopping patience steps, default is -1.
        val_check_steps (int, optional): Validation check steps, default is 100.
        batch_size (int, optional): Batch size for training, default is 32.
        valid_batch_size (int, optional): Batch size for validation, default is 32.
        windows_batch_size (int, optional): Batch size for windows, default is 32.
        inference_windows_batch_size (int, optional): Batch size for inference windows, default is 32.
        start_padding_enabled (bool, optional): If start padding is enabled, default is False.
        step_size (int, optional): Step size, default is 1.
        scaler_type (str, optional): Type of scaler, default is 'identity'.
        random_seed (int, optional): Random seed, default is 1.
        drop_last_loader (bool, optional): If the last loader should be dropped, default is False.
        optimizer (callable, optional): Optimizer, default is None.
        optimizer_kwargs (dict, optional): Optimizer keyword arguments, default is None.
        lr_scheduler (callable, optional): Learning rate scheduler, default is None.
        lr_scheduler_kwargs (dict, optional): Learning rate scheduler keyword arguments, default is None.
        dataloader_kwargs (dict, optional): Data loader keyword arguments, default is None.
        **trainer_kwargs: Additional trainer keyword arguments.
    Methods:
        forward(windows_batch):
            Forward pass of the model.
            Args:
                windows_batch (dict): Batch of windowed time series data.
            Returns:
                torch.Tensor: Weighted sum of the experts' outputs.
    """
    # Class attributes
    SAMPLING_TYPE = 'univariate'
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False

    def __init__(self,
                 h,
                 input_size,
                 dropout: float = 0.1,
                 futr_exog_list = None,
                 hist_exog_list = None,
                 stat_exog_list = None,
                 loss = SMAPE(),
                 valid_loss = None,
                 max_steps: int = 6300,
                 learning_rate: float = 1e-3,
                 num_lr_decays: int = -1,
                 early_stop_patience_steps: int =-1,
                 val_check_steps: int = 100,
                 batch_size: int = 32,
                 valid_batch_size: int = 32,
                 windows_batch_size: int = 32,
                 inference_windows_batch_size: int = 32,
                 start_padding_enabled: bool = False,
                 step_size: int = 1,
                 scaler_type: str = 'identity',
                 random_seed: int = 1,
                 drop_last_loader: bool = False,
                 optimizer = None,
                 optimizer_kwargs = None,
                 lr_scheduler = None,
                 lr_scheduler_kwargs = None,
                 dataloader_kwargs = None,
                 experts=None,
                 gate=None,
                 pooling=None,
                 aux_loss=False,
                 aux_loss_weight=1000,
                 **trainer_kwargs):
        
        if aux_loss:
            loss = AuxLoss(lambda_entropy=aux_loss_weight)
            valid_loss = AuxLoss(lambda_entropy=aux_loss_weight)

        super(SimpleMoe, self).__init__(h=h,
                                   input_size=input_size,
                                   stat_exog_list = None,
                                   futr_exog_list = None,
                                   hist_exog_list = None,
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
                                   random_seed=random_seed,
                                   drop_last_loader=drop_last_loader,
                                   optimizer=optimizer,
                                   optimizer_kwargs=optimizer_kwargs,
                                   lr_scheduler=lr_scheduler,
                                   lr_scheduler_kwargs=lr_scheduler_kwargs,
                                   dataloader_kwargs=dataloader_kwargs,
                                   **trainer_kwargs)
        

        self.input_size = input_size
        self.h = h
        self.dropout = nn.Dropout(dropout)

        if experts is not None:
            self.experts = experts
        else:
            self.experts = nn.ModuleList([
                NBEATS(self.h, self.input_size),
                NBEATS(self.h, self.input_size),
                NBEATS(self.h, self.input_size),
                NBEATS(self.h, self.input_size),
                NBEATS(self.h, self.input_size),
                # NBEATS(self.h, self.input_size),
                # NBEATS(self.h, self.input_size),
                # NBEATS(self.h, self.input_size),
                # NBEATS(self.h, self.input_size),
                # NBEATS(self.h, self.input_size),
            ])

        self.num_experts = len(self.experts)

        if gate is not None:
            self.gate = gate
        else:
            # self.gate = nn.Linear(self.input_size, self.num_experts)
            self.gate = MLP(
                self.input_size, 
                self.num_experts,
                activation="Sigmoid",
                hidden_size=64, 
                num_layers=1,
                dropout=0.1)
        self.softmax = nn.Softmax(dim=1)

        if pooling is not None:
            self.pooling = pooling
        else:
            self.pooling = SparsePooling(self.experts, self.gate, self.h)


        self.rev = RevIN(1, affine=True)

    def training_step(self, batch, batch_idx):
        # Create and normalize windows [Ws, L+H, C]
        windows = self._create_windows(batch, step="train")
        y_idx = batch["y_idx"]
        original_outsample_y = torch.clone(windows["temporal"][:, -self.h :, y_idx])
        windows = self._normalization(windows=windows, y_idx=y_idx)

        # Parse windows
        (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        ) = self._parse_windows(batch, windows)

        windows_batch = dict(
            insample_y=insample_y,  # [Ws, L]
            insample_mask=insample_mask,  # [Ws, L]
            futr_exog=futr_exog,  # [Ws, L + h, F]
            hist_exog=hist_exog,  # [Ws, L, X]
            stat_exog=stat_exog,
        )  # [Ws, S]

        # Model Predictions
        output = self(windows_batch)
        if self.loss.is_distribution_output:
            _, y_loc, y_scale = self._inv_normalization(
                y_hat=outsample_y, temporal_cols=batch["temporal_cols"], y_idx=y_idx
            )
            outsample_y = original_outsample_y
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )
            loss = self.loss(y=outsample_y, distr_args=distr_args, mask=outsample_mask)
        else:
            loss = self.loss(y=outsample_y, y_hat=output, mask=outsample_mask)
            
        probs = self.gate(windows_batch['insample_y'])
        gate_value = self.softmax(probs)
        
        one_hot_selection = torch.zeros_like(gate_value).scatter_(1, gate_value.argmax(dim=1, keepdim=True), 1)
        
        # Compute expert selection fractions f_i
        T = insample_y.shape[1]  # Time steps
        K = gate_value.shape[1]  # Number of experts
        f_i = torch.mean(one_hot_selection, dim=0)  # (Num_Experts,)

        f_i = f_i / K

        # Compute routing probabilities r_i
        r_i = torch.mean(gate_value, dim=0)  # Same shape as f_i (Num_Experts,)

        # Compute auxiliary loss
        aux_loss = torch.sum(f_i * r_i) * self.num_experts

        # Combine with primary loss
        loss = loss + aux_loss * 300
        
        # if aux_loss < 0.2:
        #     print(f"probs: {gate_value}")
            
        
        if torch.isnan(loss):
            print("Model Parameters", self.hparams)
            print("insample_y", torch.isnan(insample_y).sum())
            print("outsample_y", torch.isnan(outsample_y).sum())
            print("output", torch.isnan(output).sum())
            raise Exception("Loss is NaN, training stopped.")

        self.log(
            "train_loss",
            loss.detach().item(),
            batch_size=outsample_y.size(0),
            prog_bar=True,
            on_epoch=True,
        )
        self.train_trajectories.append((self.global_step, loss.detach().item()))
        return loss

    def forward(self, windows_batch):

        # insample_y = windows_batch['insample_y']

        # windows_batch['insample_y'] = self.rev(insample_y, "norm")
        

        # print(insample_y.shape)
        # print(insample_y)
        # print(windows_batch['insample_y'])

        # Compute the weighted sum of the experts
        
        # out1 = self.shared_expert(windows_batch)
        
        out2 = self.pooling(windows_batch)
        
        
        # gate_value = self.shared_gate(windows_batch['insample_y'])
        # out1_weighted = out1 * gate_value
        # out = out1_weighted + out2  
        
        # out = self.pooling(windows_batch)

        # gate_probs = self.softmax(self.gate(insample_y)) ## TODO:its being calculated twice, change it

        # #  # Compute entropy loss to prevent gate collapse
        # entropy_loss = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=1).mean()

        # # # Store entropy loss for later modification in loss
        # AuxLoss.current_entropy_loss = entropy_loss

        # out = self.rev(out, "denorm")

        return out2
        