

import torch
import torch.nn as nn

from neuralforecast.losses.pytorch import SMAPE
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.common._modules import RevIN, MLP
from neuralforecast.losses.pytorch import BasePointLoss

from .pooling_methods.methods import *

from neuralforecast.models.nbeats import NBEATS
from typing import Union


class SimpleMoeDLags(BaseWindows):
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
                 max_steps: int = 3000,
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

        super(SimpleMoeDLags, self).__init__(h=h,
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
                NBEATS(self.h, 18, random_seed=1, mlp_units=3 * [[64, 64]]),
                NBEATS(self.h, 24, random_seed=2, mlp_units=3 * [[64, 64]]),
                NBEATS(self.h, 36, random_seed=3, mlp_units=3 * [[64, 64]]),
                NBEATS(self.h, 54, random_seed=4, mlp_units=3 * [[64, 64]]),
                NBEATS(self.h, self.input_size, random_seed=5, mlp_units=3 * [[64, 64]]),

                # NBEATS(self.h, self.input_size, random_seed=5, mlp_units=3 * [[64, 64]]),
                # NBEATS(self.h, self.input_size),
                # NBEATS(self.h, self.input_size),
                # NBEATS(self.h, self.input_size),
                # NBEATS(self.h, self.input_size),
                # NBEATS(self.h, self.input_size),
            ])
            self.list_of_lags = [18, 24, 36, 54, self.input_size]

        self.num_experts = len(self.experts)

        if gate is not None:
            self.gate = gate
        else:
            self.gate = nn.Linear(self.input_size, self.num_experts)
            
        self.softmax = nn.Softmax(dim=1)

        if pooling is not None:
            self.pooling = pooling
        else:
            self.pooling = SparsePooling(
                self.experts, 
                self.gate,
                self.h,
                k=1,
                list_of_lags=self.list_of_lags,
            )


    def forward(self, windows_batch):

        out2 = self.pooling(windows_batch)
    
        return out2
        