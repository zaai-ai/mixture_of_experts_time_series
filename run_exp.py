import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from functools import partial

from neuralforecast import NeuralForecast
from neuralforecast.losses.numpy import smape
from datasetsforecast.m3 import M3  
from datasetsforecast.m4 import M4  
from neuralforecast.losses.pytorch import SMAPE, HuberLoss, MSE
import hydra
from omegaconf import DictConfig
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor

from torch.optim.lr_scheduler import LambdaLR, LRScheduler


import traceback

### callback
from models.callbacks.gate_distribution import GateDistributionCallback
from models.callbacks.series_distribution import SeriesDistributionCallback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.callbacks.series_similarity import SeriesSimilarityCallback


# Import your model classes (here we assume the module name matches the
# model name)
from models.SimpleMoe import SimpleMoe
from models.TimeMoeAdapted import TimeMoeAdapted
from models.InformerMoe import InformerMoe
from models.MlpMoe import MLPMoe
from neuralforecast.models import NHITS
from neuralforecast.models import NBEATS
from neuralforecast.models import VanillaTransformer
from neuralforecast.models import Autoformer
from neuralforecast.models import MLP



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
    
    # def get_cosine_schedule_with_warmup_min_lr(
    #         self,
    #         optimizer: torch.optim.Optimizer,
    #         num_warmup_steps: int,
    #         num_training_steps: int,
    #         num_cycles: float = 0.5,
    #         min_lr_ratio: float = 0,
    #         last_epoch: int = -1
    # ):
    #     lr_lambda = partial(
    #         self._get_cosine_schedule_with_warmup_and_min_lr_lambda,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=num_training_steps,
    #         num_cycles=num_cycles,
    #         min_lr_ratio=min_lr_ratio,
    #     )
    #     return LambdaLR(optimizer, lr_lambda, last_epoch)
    
    def __init__(self, optimizer,num_training_steps= 100000, num_warmup_steps = 10000 , min_lr = 0.0, last_epoch=-1, verbose=False):
       
        lr_lambda = partial(
            self._get_cosine_schedule_with_warmup_and_min_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
            min_lr_ratio=min_lr,
        )
       
        super().__init__(optimizer, lr_lambda, last_epoch)


    # def create_scheduler(optimizer, num_training_steps= 10000, num_warmup_steps = 1000 , min_lr = 1e-6):
    #     """
    #     Create a scheduler for the optimizer.
    #     """
        
    #     scheduler = get_cosine_schedule_with_warmup_min_lr(
    #         optimizer,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=num_training_steps,
    #         min_lr_ratio=min_lr,
    #     )
        
    #     return scheduler
    
    

def load_dataset(dataset_name: str, dataset_cfg: DictConfig):
    """Load dataset based on dataset_name and its configuration."""
    if dataset_name == "m3_monthly":
        print("Loading m3_monthly dataset...")
        return M3.load(
            directory=dataset_cfg.directory,
            group=dataset_cfg.group)[0]
    elif dataset_name == "m4_monthly":
        print("Loading m4_monthly dataset...")
        df = M4.load(
            directory=dataset_cfg.directory,
            group=dataset_cfg.group)[0]
        
        # Convert the 'ds' to integer
        df['ds'] = pd.to_datetime(df['ds']).astype(int)

        return df
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


def get_config_value(cfg_val, idx):
    """
    Helper: If cfg_val is a list then return the idx-th element;
    otherwise return cfg_val.
    """
    if isinstance(cfg_val, list):
        return cfg_val[idx % len(cfg_val)]
    return cfg_val


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
    elif model_name.lower() == "nbeats":
        input_size_val = get_config_value(params.input_size, config_idx)
        loss_str = get_config_value(params.loss, config_idx)
        valid_loss_str = get_config_value(params.valid_loss, config_idx)
        early_stop = get_config_value(
        params.early_stop_patience_steps, config_idx)
        batch_size_val = get_config_value(params.batch_size, config_idx)
        val_check_steps = get_config_value(params.val_check_steps, config_idx)

        model_instance = NBEATS(
            h=horizon,
            input_size=input_size_val,
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
        
        model_instance = MLPMoe(
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

    else:
        raise NotImplementedError(f"Model '{model_name}' is not implemented.")  
    return model_instance, checkpoint_callback


def calculate_smape(Y_test_df, Y_hat_df, forecast_col):
    """Calculate the sMAPE."""
    y_true = Y_test_df['y'].values
    try:
        y_hat = Y_hat_df[forecast_col].values
    except KeyError as exc:
        raise KeyError(
            f"Forecast column '{forecast_col}' not found in predictions!") from exc
   
    n_series = Y_test_df['unique_id'].nunique()
    try:
        y_true = y_true.reshape(n_series, -1)
        y_hat = y_hat.reshape(n_series, -1)
    except Exception as e:
        raise ValueError("Error reshaping arrays") from e

    return smape(y_true, y_hat)

def plot_history(history, model_name):
    print(history)
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['ptl/val_loss'], label='Validation Loss')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title(f"Training and Validation Loss ({model_name})", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.show()

def run_model_experiment(
    model_name: str,
    model_config: DictConfig,
    Y_train_df: pd.DataFrame,
    Y_test_df: pd.DataFrame,
    horizon: int,
    freq: str,
    config_idx: int = 0
):
    """
    For a given model configuration, initialize the model, run forecast, and compute sMAPE.
    Returns the computed sMAPE, the forecasts, and the model instance.
    """

    # Initialize model instance based on model_name.
    model_instance, checkpoint_callback = get_instance(
        model_name, model_config, horizon, config_idx, training_df=Y_train_df)

    # Instantiate NeuralForecast and run forecast.
    fcst = NeuralForecast(models=[model_instance], freq=freq)
    fcst.fit(df=Y_train_df, static_df=None, val_size=horizon)
    
    # fcst = SimpleMoe.load_from_checkpoint(checkpoint_callback.best_model_path + 'checkpoints/best_model.ckpt')
    # fcst = NeuralForecast(models=[fcstModel], freq=freq)
    forecasts = fcst.predict(futr_df=Y_test_df)

    
    # plot the training loss and validation loss
    # plot_history(..., model_name)

    # Extract forecast column (assumes column is named after model class)
    forecast_col = model_instance.__class__.__name__
    Y_hat_df = forecasts.reset_index(
        drop=False).drop(
        columns=[
            'unique_id',
            'ds'])

    # Evaluate sMAPE.
    current_smape = calculate_smape(Y_test_df, Y_hat_df, forecast_col)

    return current_smape, forecasts, model_instance


def plot_mean_smape(horizons, results, dataset_name: str):
    """Plot the mean sMAPE versus forecast horizons."""
    pivot_table = results.pivot_table(values='smape', index='horizon', columns='model')
    plt.figure(figsize=(10, 6))
    for model in pivot_table.columns:
        plt.plot(pivot_table.index, pivot_table[model], marker='o', linestyle='-', label=model)
    plt.xlabel("Forecast Horizon", fontsize=14)
    plt.ylabel("Mean sMAPE", fontsize=14)
    plt.title(f"Mean sMAPE vs. Forecast Horizon ({dataset_name})", fontsize=16)
    plt.grid(True)
    plt.xticks(horizons)
    plt.legend(title='Model')
    plt.savefig(f"mean_smape_vs_horizon_{dataset_name}.png")
    plt.show()


def plot_forecasts(
        Y_train_df,
        Y_test_df,
        forecasts,
        model_instance,
        dataset_name: str,
        plot_cfg: DictConfig):
    """
    Plot the ground truth and the forecasts in the same graph.
    Ground truth will be a solid line, and forecasts will be a dashed line.
    """
    forecast_col = model_instance.__class__.__name__
    # Ensure forecasts has the 'unique_id', 'ds', and forecast_col columns
    forecasts = forecasts.reset_index(drop=False)

    # Keep last 24 months from train before concatenating
    train_last_24 = Y_train_df.groupby('unique_id').tail(256)

    # Combine training and test data for the ground truth
    gt_df = pd.concat([train_last_24, Y_test_df], axis=0)
    forecasts = pd.concat([train_last_24, forecasts], axis=0)

    # Create a figure
    plt.figure(figsize=(plot_cfg.figsize[0], plot_cfg.figsize[1]))

    # Plot only the last 10 series
    unique_ids = gt_df['unique_id'].unique()[-10:]

    # Plot ground truth and forecasts
    for unique_id in unique_ids:
        temp_gt_df = gt_df[gt_df['unique_id'] == unique_id]
        temp_fcst_df = forecasts[forecasts['unique_id'] == unique_id]
        color = np.random.rand(3,)
        plt.plot(
            temp_gt_df['ds'],
            temp_gt_df['y'],
            label=f"Series {unique_id} - Ground Truth",
            color=color)
        plt.plot(
            temp_fcst_df['ds'],
            temp_fcst_df[forecast_col],
            linestyle='--',
            label=f"Series {unique_id} - Forecast",
            color=color)

    plt.title(plot_cfg.title)
    plt.xlabel(plot_cfg.xlabel)
    plt.ylabel(plot_cfg.ylabel)
    plt.grid(plot_cfg.grid)
    plt.legend(fontsize=plot_cfg.legend_font_size)

    # Save the figure
    plt.savefig(f"{dataset_name}_{plot_cfg.save_path}")
    plt.show()


def run_exp(cfg: DictConfig):
    """Run the experiment based on the configuration."""

    results = pd.DataFrame(columns=['model', 'horizon', 'smape'])

    # Loop over each active dataset specified in the config.
    for dataset_name in cfg.active_datasets:
        print(f"\n==== Running experiment for dataset: {dataset_name} ====")
        dataset_cfg = cfg.data.get(dataset_name)
        if dataset_cfg is None:
            print(
                f"Dataset configuration '{dataset_name}' not found in config.data!")
            continue

        try:
            Y_ALL = load_dataset(dataset_name, dataset_cfg)
        except Exception as e:
            print(e)
            continue

        horizons = dataset_cfg.horizons

        # Loop over each forecast horizon.
        for horizon in horizons:
            print(f"\n--- Evaluating for horizon: {horizon} ---")
            Y_train_df, Y_test_df = train_test_split(Y_ALL, horizon=horizon)
            smape_list = []

            # Loop over each active model.
            for model_name in cfg.active_models:
                print(f"\n>>> Running model: {model_name}")
                model_config = cfg.models.get(model_name)
                if model_config is None:
                    print(
                        f"Model configuration '{model_name}' not found in cfg.models!")
                    continue

                # determine the index of the model configuration
                i = horizons.index(horizon)

                try:
                    current_smape, forecasts, model_instance = run_model_experiment(
                        model_name,
                        model_config,
                        Y_train_df,
                        Y_test_df,
                        horizon,
                        dataset_cfg.freq,
                        config_idx=i
                    )
                    print(
                        f"Model '{model_name}' config {i}: sMAPE = {current_smape:.3f}")
                    smape_list.append(current_smape)
                    results = pd.concat([results,pd.DataFrame({
                        'model': [model_name],
                        'horizon': [horizon],
                        'smape': current_smape
                    })], ignore_index=True)
                except Exception as e:
                    print(
                        f"Error running model '{model_name}' config {i}: {e}")
                    traceback.print_exc()
                    continue

            if smape_list:
                mean_smape = np.mean(smape_list)
                print(f"Mean sMAPE for horizon {horizon}: {mean_smape:.3f}")
            else:
                smape_list.append(np.nan)
                print(f"No valid model runs for horizon {horizon}.")

            # Optionally, plot the forecasts from the last experiment run.
            try:
                plot_forecasts(
                    Y_train_df,
                    Y_test_df,
                    forecasts,
                    model_instance,
                    dataset_name,
                    cfg.plot.default_plot)
            except Exception as e:
                print(f"Error plotting forecasts: {e}")
        
        # Plot mean sMAPE vs. horizons.
        plot_mean_smape(horizons, results, dataset_name)

        return results, forecasts



@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    run_exp(cfg)


if __name__ == "__main__":
    main()
