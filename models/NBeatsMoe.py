
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna

from neuralforecast.losses.pytorch import MAE
from neuralforecast.common._base_windows import BaseWindows


from .pooling_methods import SparsePooling, SharedExpertPooling

class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, out_features: int = 1):
        super().__init__()
        self.out_features = out_features
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, : self.backcast_size]
        forecast = theta[:, self.backcast_size :]
        forecast = forecast.reshape(len(forecast), -1, self.out_features)
        return backcast, forecast


class TrendBasis(nn.Module):
    def __init__(
        self,
        degree_of_polynomial: int,
        backcast_size: int,
        forecast_size: int,
        out_features: int = 1,
    ):
        super().__init__()
        self.out_features = out_features
        polynomial_size = degree_of_polynomial + 1
        self.backcast_basis = nn.Parameter(
            torch.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(backcast_size, dtype=float) / backcast_size, i
                        )[None, :]
                        for i in range(polynomial_size)
                    ]
                ),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.forecast_basis = nn.Parameter(
            torch.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(forecast_size, dtype=float) / forecast_size, i
                        )[None, :]
                        for i in range(polynomial_size)
                    ]
                ),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        polynomial_size = self.forecast_basis.shape[0]  # [polynomial_size, L+H]
        backcast_theta = theta[:, :polynomial_size]
        forecast_theta = theta[:, polynomial_size:]
        forecast_theta = forecast_theta.reshape(
            len(forecast_theta), polynomial_size, -1
        )
        backcast = torch.einsum("bp,pt->bt", backcast_theta, self.backcast_basis)
        forecast = torch.einsum("bpq,pt->btq", forecast_theta, self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(
        self,
        harmonics: int,
        backcast_size: int,
        forecast_size: int,
        out_features: int = 1,
    ):
        super().__init__()
        self.out_features = out_features
        frequency = np.append(
            np.zeros(1, dtype=float),
            np.arange(harmonics, harmonics / 2 * forecast_size, dtype=float)
            / harmonics,
        )[None, :]
        backcast_grid = (
            -2
            * np.pi
            * (np.arange(backcast_size, dtype=float)[:, None] / forecast_size)
            * frequency
        )
        forecast_grid = (
            2
            * np.pi
            * (np.arange(forecast_size, dtype=float)[:, None] / forecast_size)
            * frequency
        )

        backcast_cos_template = torch.tensor(
            np.transpose(np.cos(backcast_grid)), dtype=torch.float32
        )
        backcast_sin_template = torch.tensor(
            np.transpose(np.sin(backcast_grid)), dtype=torch.float32
        )
        backcast_template = torch.cat(
            [backcast_cos_template, backcast_sin_template], dim=0
        )

        forecast_cos_template = torch.tensor(
            np.transpose(np.cos(forecast_grid)), dtype=torch.float32
        )
        forecast_sin_template = torch.tensor(
            np.transpose(np.sin(forecast_grid)), dtype=torch.float32
        )
        forecast_template = torch.cat(
            [forecast_cos_template, forecast_sin_template], dim=0
        )

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        harmonic_size = self.forecast_basis.shape[0]  # [harmonic_size, L+H]
        backcast_theta = theta[:, :harmonic_size]
        forecast_theta = theta[:, harmonic_size:]
        forecast_theta = forecast_theta.reshape(len(forecast_theta), harmonic_size, -1)
        backcast = torch.einsum("bp,pt->bt", backcast_theta, self.backcast_basis)
        forecast = torch.einsum("bpq,pt->btq", forecast_theta, self.forecast_basis)
        return backcast, forecast

# %% ../../nbs/models.nbeats.ipynb 8
ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]


class NBEATSMoEBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size: int,
        n_theta: int,
        mlp_units: list,
        basis: nn.Module,
        dropout_prob: float,
        activation: str,
        gate_input_size: int = None,
        gate_type: str = "linear",
        nr_experts: int = 8,
        top_k: int = 2,
        pre_experts: Optional[nn.ModuleList] = None,
        return_gate_logits: bool = False,
        share_experts: bool = False,
        bias_load_balancer: bool = False,
        scale_expert_complexity: bool = False,
    ):
        """ """
        super().__init__()

        self.dropout_prob = dropout_prob

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        activ = getattr(nn, activation)()

        self.nr_experts = nr_experts + 1 if share_experts else nr_experts
        self.k = top_k
        self.experts = nn.ModuleList()
        self.return_gate_logits = return_gate_logits
        self.share_experts = share_experts

        if gate_input_size is None:
            gate_input_size = input_size

        if gate_type == "linear":
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_input_size),
                nn.Linear(in_features=gate_input_size, out_features=self.nr_experts, bias=False),
            )
        elif gate_type == "mlp":
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_input_size),
                nn.Linear(in_features=gate_input_size, out_features=gate_input_size*2),
                activ,
                nn.Linear(in_features=gate_input_size*2, out_features=self.nr_experts),
            )
        elif gate_type == "conv1d-flatten":
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_input_size),
                nn.Unflatten(1, (gate_input_size, 1)),  # [batch, features] → [batch, features, 1]
                nn.Conv1d(in_channels=gate_input_size, out_channels=nr_experts, kernel_size=1),
                nn.Flatten(1),  # [batch, nr_experts, 1] → [batch, nr_experts]
            )
        elif gate_type == "conv1d-aap":
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_input_size),
                nn.Unflatten(1, (1, gate_input_size)),
                # conv block (→ [batch, 32, input_size])
                nn.Conv1d(in_channels=1,  out_channels=nr_experts*8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=nr_experts*8, out_channels=nr_experts*16, kernel_size=3, padding=1),
                nn.ReLU(),
                # global pooling → [batch, 32, 1]
                nn.AdaptiveAvgPool1d(1),
                # project to nr_experts → [batch, nr_experts, 1]
                nn.Conv1d(in_channels=nr_experts*16, out_channels=nr_experts, kernel_size=1),
                # flatten away the length dim → [batch, nr_experts]
                nn.Flatten(1),
            )
        elif gate_type == "conv1d-nopooling": # TODO: test
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_input_size),
                nn.Unflatten(1, (1, gate_input_size)),
                # conv block (→ [batch, 32, input_size])
                nn.Conv1d(in_channels=1,  out_channels=nr_experts*8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=nr_experts*8, out_channels=nr_experts*16, kernel_size=3, padding=1),
                nn.ReLU(),
                # project to nr_experts → [batch, nr_experts, 1]
                nn.Conv1d(in_channels=nr_experts*16, out_channels=nr_experts, kernel_size=1),
                # flatten away the length dim → [batch, nr_experts]
                nn.Flatten(1),
            )
        elif gate_type == "conv1d-maxpool":
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_input_size),
                nn.Unflatten(1, (1, gate_input_size)),
                nn.Conv1d(1, nr_experts * 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(nr_experts * 8, nr_experts * 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Conv1d(nr_experts * 16, nr_experts, 1),
                nn.Flatten(1),
            )
        else:
            raise ValueError(f"gate_type {gate_type} not found!")

        self.softmax = nn.Softmax(dim=1)
        self.bias_load_balancer = bias_load_balancer

        for i in range(self.nr_experts):

            if scale_expert_complexity:
                mlp_units = [
                    [int(np.ceil(layer[0] / (i+1))), int(np.ceil(layer[1]/(i+1)))] for layer in mlp_units
                ]


            hidden_layers = [
                nn.Linear(in_features=input_size, out_features=mlp_units[0][0], bias=False)
            ]
            for layer in mlp_units:
                hidden_layers.append(nn.Linear(in_features=layer[0], out_features=layer[1], bias=False))
                hidden_layers.append(activ)

                if self.dropout_prob > 0:
                    raise NotImplementedError("dropout")
                    # hidden_layers.append(nn.Dropout(p=self.dropout_prob))

            output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta, bias=False)]
            layers = hidden_layers + output_layer
            self.layers = nn.Sequential(*layers)

            self.experts.append(self.layers)

        if pre_experts is not None:
            self.experts = pre_experts

        if not self.share_experts:
            self.pooling = SparsePooling(
                experts=self.experts, gate=self.gate, out_features=n_theta, k=self.k, 
                unpack=False, return_soft_gates=True,
                bias=self.bias_load_balancer,
            )
        else:
            self.pooling = SharedExpertPooling(
                experts=self.experts[1:],
                sparse_gate=self.gate,
                gate= nn.Linear(
                    in_features=input_size,
                    out_features=2
                ),
                shared_expert=self.experts[0],
                out_features=n_theta, k=self.k, 
                unpack=False, return_soft_gates=True,
                bias=self.bias_load_balancer,
            )

        del self.layers
        del self.experts

        self.n_theta = n_theta
        self.basis = basis

    def forward(self, insample_y: torch.Tensor, gate_insample: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        theta, gate_logits = self.pooling(insample_y, gate_insample=gate_insample)
        
        backcast, forecast = self.basis(theta)

        if self.return_gate_logits:
             return backcast, forecast, gate_logits

        return backcast, forecast

# %% ../../nbs/models.nbeats.ipynb 9
class NBeatsMoe(BaseWindows):
    """NBEATS

    The Neural Basis Expansion Analysis for Time Series (NBEATS), is a simple and yet
    effective architecture, it is built with a deep stack of MLPs with the doubly
    residual connections. It has a generic and interpretable architecture depending
    on the blocks it uses. Its interpretable architecture is recommended for scarce
    data settings, as it regularizes its predictions through projections unto harmonic
    and trend basis well-suited for most forecasting tasks.

    **Parameters:**<br>
    `h`: int, forecast horizon.<br>
    `input_size`: int, considered autorregresive inputs (lags), y=[1,2,3,4] input_size=2 -> lags=[1,2].<br>
    `n_harmonics`: int, Number of harmonic terms for seasonality stack type. Note that len(n_harmonics) = len(stack_types). Note that it will only be used if a seasonality stack is used.<br>
    `n_polynomials`: int, polynomial degree for trend stack. Note that len(n_polynomials) = len(stack_types). Note that it will only be used if a trend stack is used.<br>
    `stack_types`: List[str], List of stack types. Subset from ['seasonality', 'trend', 'identity'].<br>
    `n_blocks`: List[int], Number of blocks for each stack. Note that len(n_blocks) = len(stack_types).<br>
    `mlp_units`: List[List[int]], Structure of hidden layers for each stack type. Each internal list should contain the number of units of each hidden layer. Note that len(n_hidden) = len(stack_types).<br>
    `dropout_prob_theta`: float, Float between (0, 1). Dropout for N-BEATS basis.<br>
    `shared_weights`: bool, If True, all blocks within each stack will share parameters. <br>
    `activation`: str, activation from ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid'].<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `num_lr_decays`: int=3, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int=32, number of different series in each batch.<br>
    `valid_batch_size`: int=None, number of different series in each validation and test batch, if None uses batch_size.<br>
    `windows_batch_size`: int=1024, number of windows to sample in each training batch, default uses all.<br>
    `inference_windows_batch_size`: int=-1, number of windows to sample in each inference batch, -1 uses all.<br>
    `start_padding_enabled`: bool=False, if True, the model will pad the time series with zeros at the beginning, by input size.<br>
    `step_size`: int=1, step size between each window of temporal data.<br>
    `scaler_type`: str='identity', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int, random_seed for pytorch initializer and numpy generators.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `optimizer`: Subclass of 'torch.optim.Optimizer', optional, user specified optimizer instead of the default choice (Adam).<br>
    `optimizer_kwargs`: dict, optional, list of parameters used by the user specified `optimizer`.<br>
    `lr_scheduler`: Subclass of 'torch.optim.lr_scheduler.LRScheduler', optional, user specified lr_scheduler instead of the default choice (StepLR).<br>
    `lr_scheduler_kwargs`: dict, optional, list of parameters used by the user specified `lr_scheduler`.<br>
    `dataloader_kwargs`: dict, optional, list of parameters passed into the PyTorch Lightning dataloader by the `TimeSeriesDataLoader`. <br>
    `**   `: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>

    **References:**<br>
    -[Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, Yoshua Bengio (2019).
    "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting".](https://arxiv.org/abs/1905.10437)
    """

    # Class attributes
    SAMPLING_TYPE = "windows"
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False

    def __init__(
        self,
        h,
        input_size,
        n_harmonics: int = 2,
        n_polynomials: int = 2,
        stack_types: list = ["identity", "trend", "seasonality"],
        n_blocks: list = [1, 1, 1],
        mlp_units: list = 3 * [[128, 128]],
        dropout_prob_theta: float = 0.0,
        activation: str = "ReLU",
        shared_weights: bool = True,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled=False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        gate_type: str = "linear",
        nr_experts: int = 4,
        top_k: int = 2,
        pre_blocks: Optional[nn.ModuleList] = None,
        share_experts: bool = False,
        bias_load_balancer: bool = False,
        auxiliary_loss: bool = False,
        return_gate_logits: bool = True,
        drop_last_loader: bool = False,
        store_all_gate_logits: bool = True,
        use_forecast_as_input: bool = False,
        scale_expert_complexity: bool = False,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):

        # Protect horizon collapsed seasonality and trend NBEATSx-i basis
        if h == 1 and (("seasonality" in stack_types) or ("trend" in stack_types)):
            raise Exception(
                "Horizon `h=1` incompatible with `seasonality` or `trend` in stacks"
            )

        # Inherit BaseWindows class
        super(NBeatsMoe, self).__init__(
            h=h,
            input_size=input_size,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            windows_batch_size=windows_batch_size,
            valid_batch_size=valid_batch_size,
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
            **trainer_kwargs,
        )

        if top_k > nr_experts:
            raise optuna.TrialPruned(f"Check top_k={top_k} <= nr_experts={nr_experts}")# raise Exception(

        self.nr_experts = nr_experts
        self.top_k = top_k 
        self.return_gate_logits = return_gate_logits
        self._training = True
        self.store_all_gate_logits = store_all_gate_logits
        self.share_experts = share_experts
        self.bias_load_balancer = bias_load_balancer
        self.scale_expert_complexity = scale_expert_complexity
        self.gate_type = gate_type
        self.use_forecast_as_input = use_forecast_as_input
        self.auxiliary_loss = auxiliary_loss

        # Architecture
        blocks = self.create_stack(
            h=h,
            input_size=input_size,
            stack_types=stack_types,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
            dropout_prob_theta=dropout_prob_theta,
            activation=activation,
            shared_weights=shared_weights,
            n_polynomials=n_polynomials,
            n_harmonics=n_harmonics,
        )
        self.blocks = torch.nn.ModuleList(blocks) if pre_blocks is None else pre_blocks
        self.all_gate_logits = [[] for _ in range(len(blocks))]
        self.all_inputs = [[] for _ in range(len(blocks))]

        self.n_blocks = n_blocks

    def predict_step(self, batch, batch_idx):
        self._training = False
        return super().predict_step(batch, batch_idx)

    def create_stack(
        self,
        stack_types,
        n_blocks,
        input_size,
        h,
        mlp_units,
        dropout_prob_theta,
        activation,
        shared_weights,
        n_polynomials,
        n_harmonics,
    ):

        block_list = []
        for i in range(len(stack_types)):
            for block_id in range(n_blocks[i]):

                # Shared weights
                if shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    if stack_types[i] == "seasonality":
                        n_theta = (
                            2
                            * (self.loss.outputsize_multiplier + 1)
                            * int(np.ceil(n_harmonics / 2 * h) - (n_harmonics - 1))
                        )
                        basis = SeasonalityBasis(
                            harmonics=n_harmonics,
                            backcast_size=input_size,
                            forecast_size=h,
                            out_features=self.loss.outputsize_multiplier,
                        )

                    elif stack_types[i] == "trend":
                        n_theta = (self.loss.outputsize_multiplier + 1) * (
                            n_polynomials + 1
                        )
                        basis = TrendBasis(
                            degree_of_polynomial=n_polynomials,
                            backcast_size=input_size,
                            forecast_size=h,
                            out_features=self.loss.outputsize_multiplier,
                        )

                    elif stack_types[i] == "identity":
                        n_theta = input_size + self.loss.outputsize_multiplier * h
                        basis = IdentityBasis(
                            backcast_size=input_size,
                            forecast_size=h,
                            out_features=self.loss.outputsize_multiplier,
                        )
                    else:
                        raise ValueError(f"Block type {stack_types[i]} not found!")

                    nbeats_block = NBEATSMoEBlock(
                        input_size=input_size,
                        gate_input_size=h if self.use_forecast_as_input else None,
                        n_theta=n_theta,
                        mlp_units=mlp_units,
                        basis=basis,
                        dropout_prob=dropout_prob_theta,
                        activation=activation,
                        nr_experts=self.nr_experts,
                        top_k=self.top_k,
                        gate_type=self.gate_type,
                        return_gate_logits=self.return_gate_logits,
                        share_experts=self.share_experts,
                        bias_load_balancer=self.bias_load_balancer,
                        scale_expert_complexity=self.scale_expert_complexity,
                    )

                # Select type of evaluation and apply it to all layers of block
                block_list.append(nbeats_block)

        return block_list
    
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
            if self.auxiliary_loss:
                last_batch_layers = torch.cat(
                    [torch.cat(layer[-1:], dim=0) for layer in self.all_gate_logits], dim=0
                )  # shape: [batch_size, nr_experts]
                # Get top-k expert indices and their corresponding values
                top_k = self.top_k
                top_k_values, top_k_experts = torch.topk(last_batch_layers, top_k, dim=1)  # shape: [batch_size, k]

                # Flatten the expert indices
                flattened_experts = top_k_experts.flatten()
                flattened_values = top_k_values.flatten()

                # Count how many times each expert was selected (hard count)
                expert_counts = torch.zeros(self.nr_experts, device=last_batch_layers.device)
                expert_counts.scatter_add_(
                    0,
                    flattened_experts,
                    torch.ones_like(flattened_experts, dtype=expert_counts.dtype)
                )

                # Sum of gate values for each expert
                expert_value_sums = torch.zeros_like(expert_counts)
                expert_value_sums.scatter_add_(
                    0,
                    flattened_experts,
                    flattened_values
                )

                # Mean gate value per expert (only for selected times)
                expert_means = torch.zeros_like(expert_counts)
                expert_means = expert_value_sums / (expert_counts + 1e-8)  # prevent division by zero

                # Multiply mean value × how often it was chosen
                soft_expert_usage = expert_counts * expert_means

                # Normalize to get a probability distribution
                p = soft_expert_usage / soft_expert_usage.sum()

                p = p.clamp(min=1e-8)

                # Uniform distribution target
                target = torch.full_like(p, 1.0 / self.nr_experts)

                # KL divergence (with log(p))
                lb_loss = F.kl_div(p.log(), target, reduction="batchmean")

                self.log(
                    "lb_loss",
                    lb_loss.detach().item() * 10,
                    batch_size=outsample_y.size(0),
                    prog_bar=True,
                    on_epoch=True,
                )

                # Final loss
                loss = loss + lb_loss * 10

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

        # Parse windows_batch
        insample_y = windows_batch["insample_y"]
        insample_mask = windows_batch["insample_mask"]

        # NBEATS' forward
        residuals = insample_y.flip(dims=(-1,))  # backcast init
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:, None]  # Level with Naive1
        block_forecasts = [forecast.repeat(1, self.h, 1)]
        block_forecast = insample_y[:, -18:, None]
        for i, block in enumerate(self.blocks):
            if self.return_gate_logits:
                if self.use_forecast_as_input:
                    block_forecast: torch.Tensor = block_forecast[:, -self.h:, :].squeeze(-1)
                    backcast, block_forecast, gate_logits = block(insample_y=residuals, gate_insample=block_forecast)
                else:
                    backcast, block_forecast, gate_logits = block(insample_y=residuals)
                if not self._training or self.store_all_gate_logits:
                    self.all_gate_logits[i].append(gate_logits)
                    self.all_inputs[i].append(residuals)
            else:
                backcast, block_forecast = block(insample_y=residuals)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast

            if self.decompose_forecast:
                block_forecasts.append(block_forecast)

        # Adapting output's domain
        forecast = self.loss.domain_map(forecast)

        if self.decompose_forecast:
            # (n_batch, n_blocks, h, out_features)
            block_forecasts = torch.stack(block_forecasts)
            block_forecasts = block_forecasts.permute(1, 0, 2, 3)
            block_forecasts = block_forecasts.squeeze(-1)  # univariate output
            return block_forecasts
        else:
            return forecast
