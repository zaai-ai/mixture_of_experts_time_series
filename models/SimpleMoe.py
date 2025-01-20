import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralforecast.losses.pytorch import SMAPE
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.common._modules import RevIN


# experts
from neuralforecast.models.nbeats import NBEATS
from neuralforecast.models.mlp import MLP


class SimpleMoe(BaseWindows):
    
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
                 max_steps: int = 1000,
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
                 **trainer_kwargs):
        
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
        
        self.experts = nn.ModuleList([
            NBEATS(h=h,
                   input_size=input_size,
                   futr_exog_list = None,
                   hist_exog_list = None,
                   stat_exog_list = None,
                   loss = SMAPE(),
                   valid_loss = None,
                   max_steps = max_steps,
                   learning_rate = learning_rate,
                   num_lr_decays = num_lr_decays,
                   early_stop_patience_steps = early_stop_patience_steps,
                   val_check_steps = val_check_steps,
                   batch_size = batch_size,
                   step_size = step_size,
                   scaler_type = scaler_type,
                   random_seed = random_seed,
                   drop_last_loader = drop_last_loader,
                   optimizer = optimizer,
                   optimizer_kwargs = optimizer_kwargs,
                   lr_scheduler = lr_scheduler,
                   lr_scheduler_kwargs = lr_scheduler_kwargs,
                   dataloader_kwargs = dataloader_kwargs,
                   **trainer_kwargs),
            MLP(h=h,
                input_size=input_size,
                futr_exog_list = None,
                hist_exog_list = None,
                stat_exog_list = None,
                loss = SMAPE(),
                valid_loss = None,
                max_steps = max_steps,
                learning_rate = learning_rate,
                num_lr_decays = num_lr_decays,
                early_stop_patience_steps = early_stop_patience_steps,
                val_check_steps = val_check_steps,
                batch_size = batch_size,
                step_size = step_size,
                scaler_type = scaler_type,
                random_seed = random_seed,
                drop_last_loader = drop_last_loader,
                optimizer = optimizer,
                optimizer_kwargs = optimizer_kwargs,
                lr_scheduler = lr_scheduler,
                lr_scheduler_kwargs = lr_scheduler_kwargs,
                dataloader_kwargs = dataloader_kwargs,
                **trainer_kwargs),
            nn.Linear(self.input_size, self.h),
        ])
        
        self.num_experts = len(self.experts)
        self.gate = nn.Linear(self.input_size, self.num_experts)
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        # Compute the gate
        gate = self.gate(x)
        gate = self.softmax(gate)
        
        # Compute the weighted sum of the experts
        weighted_sum = torch.zeros(x.size(0), self.h)
        for i in range(self.num_experts):
            weighted_sum += gate[:, i].unsqueeze(1) * self.experts[i](x)
            
            
        
        
        return weighted_sum
        