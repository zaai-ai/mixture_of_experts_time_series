
import pandas as pd
from datasetsforecast.m3 import M3  
from datasetsforecast.m4 import M4
from omegaconf import DictConfig

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor
import math
from functools import partial

from neuralforecast.losses.pytorch import SMAPE, HuberLoss, MSE, MAPE, MAE

from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from datasets.load_data.gluonts_dataset import GluontsDataset

# Import your model classes
from models.SimpleMoe import SimpleMoe
from models.SimpleMoe_NLags import SimpleMoeDLags
from models.TimeMoeAdapted import TimeMoeAdapted
from models.InformerMoe import InformerMoe
from models.MlpMoe import MLPMoe
from models.NBeatsMoe import NBeatsMoe
from models.NBeatsMoeLags import NBeatsMoeLags
from models.NBeatsStackMoe import NBeatsStackMoe
from neuralforecast.models import NHITS
from neuralforecast.models import NBEATS
from neuralforecast.models import VanillaTransformer
from neuralforecast.models import Autoformer
from neuralforecast.models import MLP
from neuralforecast.models import TCN



### callback
from models.callbacks.gate_distribution import GateDistributionCallback
from models.callbacks.series_distribution import SeriesDistributionCallback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.callbacks.series_similarity import SeriesSimilarityCallback
from models.callbacks.probs_collector import GateValuesCollectorCallback



class WarmupWithCosineLR(LambdaLR):
    
    def _get_cosine_schedule_with_warmup_and_min_lr_lambda(
            self,
            current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_ratio: float,
    ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

        return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_ratio)
    
    def __init__(self, optimizer,num_training_steps= 100000, num_warmup_steps = 10000 , min_lr = 0.0, last_epoch=-1, verbose=False):
       
        lr_lambda = partial(
            self._get_cosine_schedule_with_warmup_and_min_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
            min_lr_ratio=min_lr,
        )
       
        super().__init__(optimizer, lr_lambda, last_epoch)



def get_config_value(cfg_val, idx):
    """
    Helper: If cfg_val is a list then return the idx-th element;
    otherwise return cfg_val.
    """
    if isinstance(cfg_val, list):
        return cfg_val[idx % len(cfg_val)]
    return cfg_val

def load_dataset(dataset_name: str, dataset_cfg: DictConfig):
    """Load dataset based on dataset_name and its configuration."""
    if dataset_name == "m3":
        print("Loading m3_monthly dataset...")
        return M3.load(
            directory=dataset_cfg["directory"],
            group=dataset_cfg["group"])[0]
    elif dataset_name == "m4":
        print("Loading m4_monthly dataset...")
        df = M4.load(
            directory=dataset_cfg["directory"],
            group=dataset_cfg["group"])[0]
        
        # Convert the 'ds' to integer
        df['ds'] = pd.to_datetime(df['ds']).astype(int)

        return df
    elif dataset_name.startswith("gluonts_"):
        group = dataset_name.replace("gluonts_", "")
        print(f"Loading {group} dataset...")
        df, horizon, n_lags, freq_str, freq_int = GluontsDataset.load_everything(group)

        df['y'] = df['y'].astype(float)

        return df, horizon, n_lags, freq_str, freq_int
    else:
        raise ValueError(
            f"Loading method for dataset '{dataset_name}' is not defined.")

def train_test_split(df: pd.DataFrame, horizon: int):
    """Split the dataframe into training and test sets by horizon."""
    groups = df.groupby('unique_id')
    train_list, test_list = [], []
    for _, group_df in groups:
        group_df = group_df.sort_values('ds')
        train_list.append(group_df.head(-horizon))
        test_list.append(group_df.tail(horizon))
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df, test_df

def get_instance(
        model_name: str,
        model_config: DictConfig,
        horizon: int,
        config_idx: int = 0,
        **kwargs):
    """
    For a given model configuration, initialize the model instance.
    Returns the model instance.
    """
    params = model_config.params
    checkpoint_callback = ModelCheckpoint(
        monitor="ptl/val_loss",
        mode="min", 
        save_top_k=1,
        verbose=True,
        filename="best_model",
        dirpath="checkpoints/",
    )

    callbacks = []

    # Initialize model instance based on model_name.
    if model_name.lower() == "simplemoe":
        input_size_val = get_config_value(params.input_size, config_idx)
        dropout_val = get_config_value(params.dropout, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
            params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)

        model_instance = SimpleMoe(
            h=horizon,
            input_size=input_size_val,
            dropout=dropout_val,
            # e.g., eval("SMAPE")() creates an instance of SMAPE.
            loss=eval(valid_loss_str)(),
            valid_loss=eval(valid_loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            enable_checkpointing=True,
            val_check_steps=val_check_steps,
            # scaler_type='standard',
            # callbacks= [ SeriesDistributionCallback(**kwargs)], # GateDistributionCallback(**kwargs)
            # scaler_type='minmax',     
            # callbacks=[LearningRateMonitor(logging_interval='step')],
            callbacks= [ checkpoint_callback] #, SeriesSimilarityCallback(**kwargs) ]#SeriesDistributionCallback(**kwargs)], # GateDistributionCallback(**kwargs)
    )
    elif model_name.lower() == "simplemoe_dlags":
        input_size_val = get_config_value(params.input_size, config_idx)
        dropout_val = get_config_value(params.dropout, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
            params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)

        model_instance = SimpleMoeDLags(
            h=horizon,
            input_size=input_size_val,
            dropout=dropout_val,
            loss=eval(valid_loss_str)(),
            valid_loss=eval(valid_loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            enable_checkpointing=True,
            val_check_steps=val_check_steps,
            # callbacks= [ checkpoint_callback] 
    )
    elif model_name.lower() == "nbeats":
        input_size_val = get_config_value(params.input_size, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
        params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)

        num_training_steps = 10000

        model_instance = NBEATS(
            h=horizon,
            input_size=input_size_val,
            max_steps=num_training_steps,
            loss=eval(loss_str)(),
            valid_loss=eval(loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            # callbacks=[checkpoint_callback],
            enable_checkpointing=True,
            val_check_steps=val_check_steps,
            # scaler_type='standard',
        )
    elif model_name.lower() == "nhits":
        input_size_val = get_config_value(params.input_size, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
            params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)

        model_instance = NHITS(
            h=horizon,
            input_size=input_size_val,
            loss=eval(loss_str)(),
            valid_loss=eval(valid_loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            callbacks=[checkpoint_callback],
            enable_checkpointing=True,
            val_check_steps=val_check_steps,
            # scaler_type='standard',
        )
    elif model_name.lower() == "timemoeadapted":
        input_size_val = get_config_value(params.input_size, config_idx)
        dropout_val = get_config_value(params.dropout, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
            params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)
        
        optimizer = torch.optim.AdamW
        num_training_steps = 10000
        
        model_instance = TimeMoeAdapted(
            h=horizon,
            input_size=input_size_val,
            dropout=dropout_val,
            loss=eval(loss_str)(),
            valid_loss=eval(valid_loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            enable_checkpointing=True,
            max_steps=num_training_steps,
            optimizer=optimizer,
            optimizer_kwargs={'lr': 1e-3, 'weight_decay': 0.1, 'betas': (0.9, 0.95)},
            lr_scheduler=WarmupWithCosineLR,
            lr_scheduler_kwargs={
                'num_training_steps': num_training_steps,
                'num_warmup_steps': 1000, 
                'min_lr': 1e-6
                },
            val_check_steps=val_check_steps,
            # callbacks= [ checkpoint_callback, SeriesSimilarityCallback(**kwargs) ]#SeriesDistributionCallback(**kwargs)], # GateDistributionCallback(**kwargs)
        )
    elif model_name.lower() == "vanillatransformer":
        input_size_val = get_config_value(params.input_size, config_idx)
        dropout_val = get_config_value(params.dropout, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
            params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)
        
        optimizer = torch.optim.AdamW
        num_training_steps = 10000
        
        model_instance = VanillaTransformer(
            h=horizon,
            input_size=input_size_val,
            dropout=dropout_val,
            loss=eval(loss_str)(),
            valid_loss=eval(valid_loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            enable_checkpointing=True,
            max_steps=num_training_steps,
            scaler_type='minmax',
            optimizer=optimizer,
            optimizer_kwargs={'lr': 1e-3, 'weight_decay': 0.1, 'betas': (0.9, 0.95)},
            # lr_scheduler=WarmupWithCosineLR,
            # lr_scheduler_kwargs={
            #     'num_training_steps': num_training_steps,
            #     'num_warmup_steps': 1000, 
            #     'min_lr': 1e-6
            #     },
            val_check_steps=val_check_steps,
            # callbacks= [ checkpoint_callback, SeriesSimilarityCallback(**kwargs) ]#SeriesDistributionCallback(**kwargs)], # GateDistributionCallback(**kwargs)
        )
    elif model_name.lower() == "autoformer":
        input_size_val = get_config_value(params.input_size, config_idx)
        dropout_val = get_config_value(params.dropout, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
            params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)
        
        optimizer = torch.optim.AdamW
        num_training_steps = 10000
        
        model_instance = Autoformer(
            h=horizon,
            input_size=input_size_val,
            dropout=dropout_val,
            loss=eval(loss_str)(),
            valid_loss=eval(valid_loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            enable_checkpointing=True,
            max_steps=num_training_steps,
            scaler_type='standard',
            optimizer=optimizer,
            optimizer_kwargs={'lr' : 1e-3, 'weight_decay': 0.1, 'betas': (0.9, 0.95)},
            # lr_scheduler=WarmupWithCosineLR,
            # lr_scheduler_kwargs={
            #     'num_training_steps': num_training_steps,
            #     'num_warmup_steps': 1000, 
            #     'min_lr': 1e-6
            #     },
            hidden_size=128,
            n_head=8,
            windows_batch_size=256,
            val_check_steps=val_check_steps,
            # callbacks= [ checkpoint_callback, SeriesSimilarityCallback(**kwargs) ]#SeriesDistributionCallback(**kwargs)], # GateDistributionCallback(**kwargs)
        )
    elif model_name.lower() == "informermoe":
        input_size_val = get_config_value(params.input_size, config_idx)
        dropout_val = get_config_value(params.dropout, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
            params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)
        
        optimizer = torch.optim.AdamW
        num_training_steps = 10000
        
        model_instance = InformerMoe(
            h=horizon,
            input_size=input_size_val,
            dropout=dropout_val,
            loss=eval(loss_str)(),
            valid_loss=eval(valid_loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            enable_checkpointing=True,
            max_steps=num_training_steps,
            scaler_type='minmax',
            optimizer=optimizer,
            optimizer_kwargs={'lr': 1e-3, 'weight_decay': 0.1, 'betas': (0.9, 0.95)},
            # lr_scheduler=WarmupWithCosineLR,
            # lr_scheduler_kwargs={
            #     'num_training_steps': num_training_steps,
            #     'num_warmup_steps': 1000, 
            #     'min_lr': 1e-6
            #     },
            val_check_steps=val_check_steps,
            # callbacks= [ checkpoint_callback, SeriesSimilarityCallback(**kwargs) ]#SeriesDistributionCallback(**kwargs)], # GateDistributionCallback(**kwargs)
        )
    elif model_name.lower() == "tcn":
        input_size_val = get_config_value(params.input_size, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
            params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)

        model_instance = TCN(
            h=horizon,
            input_size=input_size_val,
            loss=eval(loss_str)(),
            valid_loss=eval(valid_loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            # callbacks=[checkpoint_callback],
            enable_checkpointing=True,
            val_check_steps=val_check_steps,
            # scaler_type='standard',
        )
    elif model_name.lower() == "mlp":

        input_size_val = get_config_value(params.input_size, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
            params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)
        
        optimizer = torch.optim.AdamW
        num_training_steps = 10000
        
        model_instance = MLP(
            h=horizon,
            input_size=input_size_val,
            loss=eval(loss_str)(),
            valid_loss=eval(valid_loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            enable_checkpointing=True,
            max_steps=num_training_steps,
            hidden_size=4096,
            scaler_type='minmax',
            optimizer=optimizer,
            optimizer_kwargs={'lr': 1e-3, 'weight_decay': 0.1, 'betas': (0.9, 0.95)},
            # lr_scheduler=WarmupWithCosineLR,
            # lr_scheduler_kwargs={
            #     'num_training_steps': num_training_steps,
            #     'num_warmup_steps': 1000, 
            #     'min_lr': 1e-6
            #     },
            val_check_steps=val_check_steps,
            # callbacks= [ checkpoint_callback, SeriesSimilarityCallback(**kwargs) ]#SeriesDistributionCallback(**kwargs)], # GateDistributionCallback(**kwargs)
        )
    elif model_name.lower() == "mlpmoe":

        input_size_val = get_config_value(params.input_size, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
            params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)
        
        optimizer = torch.optim.AdamW
        num_training_steps = 10000
        
        model_instance = MLPMoe(
            h=horizon,
            input_size=input_size_val,
            loss=eval(loss_str)(),
            valid_loss=eval(valid_loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            enable_checkpointing=True,
            max_steps=2900,
            hidden_size=4096,
            scaler_type='minmax',
            optimizer=optimizer,
            optimizer_kwargs={'lr': 1e-3, 'weight_decay': 0.1, 'betas': (0.9, 0.95)},
            lr_scheduler=WarmupWithCosineLR,
            lr_scheduler_kwargs={
                'num_training_steps': num_training_steps,
                'num_warmup_steps': 1000, 
                'min_lr': 1e-6
                },
            val_check_steps=val_check_steps,
            # callbacks= [ checkpoint_callback]#, SeriesSimilarityCallback(**kwargs) ]#SeriesDistributionCallback(**kwargs)], # GateDistributionCallback(**kwargs)
        )
    elif model_name.lower() == "nbeatsmoe":
        input_size_val = get_config_value(params.input_size, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
        params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)
        prob_collector = GateValuesCollectorCallback()

        num_training_steps = 10000

        model_instance = NBeatsMoe(
            h=horizon,
            input_size=input_size_val,
            max_steps=num_training_steps,
            loss=eval(loss_str)(),
            valid_loss=eval(loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            callbacks=[prob_collector],
            return_gate_logits=True,
            enable_checkpointing=True,
            val_check_steps=val_check_steps,
            # scaler_type='standard',
        )

        callbacks.append(prob_collector)
    elif model_name.lower() == "nbeatsmoelags":
        input_size_val = get_config_value(params.input_size, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
        params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)
        prob_collector = GateValuesCollectorCallback()

        num_training_steps = 10000

        model_instance = NBeatsMoeLags(
            h=horizon,
            input_size=input_size_val,
            max_steps=num_training_steps,
            loss=eval(loss_str)(),
            valid_loss=eval(loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            callbacks=[prob_collector],
            return_gate_logits=True,
            enable_checkpointing=True,
            val_check_steps=val_check_steps,
            # scaler_type='standard',
        )

        callbacks.append(prob_collector)
    elif model_name.lower() == "nbeatsstackmoe":
        input_size_val = get_config_value(params.input_size, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
        params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)
        prob_collector = GateValuesCollectorCallback(is_stack=True)
        

        num_training_steps = 10000

        model_instance = NBeatsStackMoe(
            h=horizon,
            input_size=input_size_val,
            max_steps=num_training_steps,
            loss=eval(loss_str)(),
            valid_loss=eval(loss_str)(),
            early_stop_patience_steps=early_stop,
            batch_size=batch_size_val,
            # callbacks=[checkpoint_callback],
            callbacks=[prob_collector],
            enable_checkpointing=True,
            val_check_steps=val_check_steps,
            # scaler_type='standard',
        )
    else:
        raise NotImplementedError(f"Model '{model_name}' is not implemented.")  
    return model_instance, callbacks

