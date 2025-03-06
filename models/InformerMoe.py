from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.losses.pytorch import MAE
from models.pooling_methods.methods import SparsePooling
import math

from neuralforecast.common._modules import (
    TransEncoderLayer,
    TransEncoder,
    TransDecoderLayer,
    TransDecoder,
    DataEmbedding,
    AttentionLayer,
)

# %% ../../nbs/models.vanillatransformer.ipynb 8
class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    """
    FullAttention
    """

    def __init__(
        self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class DenseMoe(nn.Module):

    def __init__(self, experts, gate, out_features, k=1):
        super(SparseMoe, self).__init__()
        self.experts = experts
        self.gate = gate
        self.out_features = out_features
        self.top_k = k
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, insample_y):
        gate_logits = self.gate(insample_y)
        gate_probs = self.softmax(gate_logits)
        weighted_sum = torch.zeros(insample_y.size(0), insample_y.size(1), self.out_features, device=insample_y.device)

        

        for i, expert in enumerate(self.experts):
            weighted_sum += gate_probs[:, :, i].unsqueeze(2) * expert.to(insample_y.device)(insample_y)
        return weighted_sum

class SparseMoe(nn.Module):

    def __init__(self, experts, gate, out_features, k=1):
        super(SparseMoe, self).__init__()
        self.experts = experts
        self.gate = gate
        self.out_features = out_features
        self.top_k = k
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (batch, time, features) for example
        gate_logits = self.gate(x)  # shape: (batch, time, num_experts)
        gate_probs = self.softmax(gate_logits)
        
        # Get top-k probabilities and corresponding expert indices
        routing_weights, selected_experts = torch.topk(gate_probs, self.top_k, dim=-1)
        # routing_weights & selected_experts shape: (batch, time, k)
        
        # Create one-hot mask: shape (batch, time, k, num_experts)
        expert_mask = F.one_hot(selected_experts, num_classes=len(self.experts))
        
        # Initialize weighted sum
        weighted_sum = torch.zeros(x.size(0), x.size(1), self.out_features, device=x.device)
        
        # For each expert, sum its contributions across the top-k selections
        for i, expert in enumerate(self.experts):
            # Get the mask for expert i: shape (batch, time, k)
            mask = expert_mask[..., i].float()
            # Combine mask with routing weights and sum over the k dimension.
            expert_routing = (routing_weights * mask).sum(dim=-1)  # shape: (batch, time)
            
            # Compute expert's output: assume output shape (batch, time, out_features)
            expert_output = expert.to(x.device)(x)
            # Weight the expert's output by the corresponding routing weights.
            weighted_sum += expert_output * expert_routing.unsqueeze(-1)
            
        return weighted_sum


class InformerMoe(BaseWindows):
   
    # Class attributes
    SAMPLING_TYPE = "windows"
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False

    def __init__(
        self,
        h: int,
        input_size: int,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        decoder_input_size_multiplier: float = 0.5,
        hidden_size: int = 128,
        dropout: float = 0.05,
        n_head: int = 4,
        conv_hidden_size: int = 32,
        activation: str = "gelu",
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 5000,
        learning_rate: float = 1e-4,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size=1024,
        inference_windows_batch_size: int = 1024,
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
        **trainer_kwargs,
    ):
        super(InformerMoe, self).__init__(
            h=h,
            input_size=input_size,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            futr_exog_list=futr_exog_list,
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
            **trainer_kwargs,
        )

        # Architecture
        self.label_len = int(np.ceil(input_size * decoder_input_size_multiplier))
        if (self.label_len >= input_size) or (self.label_len <= 0):
            raise Exception(
                f"Check decoder_input_size_multiplier={decoder_input_size_multiplier}, range (0,1)"
            )

        if activation not in ["relu", "gelu"]:
            raise Exception(f"Check activation={activation}")

        self.c_out = self.loss.outputsize_multiplier
        self.output_attention = False
        self.enc_in = 1
        self.dec_in = 1

        # Embedding
        self.enc_embedding = DataEmbedding(
            c_in=self.enc_in,
            exog_input_size=self.futr_exog_size,
            hidden_size=hidden_size,
            pos_embedding=True,
            dropout=dropout,
        )
        self.dec_embedding = DataEmbedding(
            self.dec_in,
            exog_input_size=self.futr_exog_size,
            hidden_size=hidden_size,
            pos_embedding=True,
            dropout=dropout,
        )

        # Encoder
        self.encoder = TransEncoder(
            [
                TransEncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=self.output_attention,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size,
                    conv_hidden_size,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(encoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_size),
        )
        # Decoder
        self.decoder = TransDecoder(
            [
                TransDecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=True,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size,
                    conv_hidden_size,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(decoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_size),
            # projection=nn.Linear(hidden_size, self.c_out, bias=True),
            projection=SparseMoe(
                experts=[nn.Linear(hidden_size, self.c_out, bias=True) for _ in range(8)],
                gate=nn.Linear(hidden_size, 8, bias=True),
                out_features=self.c_out,
                k=1,
            ),
        )

    def forward(self, windows_batch):
        # Parse windows_batch
        insample_y = windows_batch["insample_y"]
        # insample_mask = windows_batch['insample_mask']
        # hist_exog     = windows_batch['hist_exog']
        # stat_exog     = windows_batch['stat_exog']

        futr_exog = windows_batch["futr_exog"]

        insample_y = insample_y.unsqueeze(-1)  # [Ws,L,1]

        if self.futr_exog_size > 0:
            x_mark_enc = futr_exog[:, : self.input_size, :]
            x_mark_dec = futr_exog[:, -(self.label_len + self.h) :, :]
        else:
            x_mark_enc = None
            x_mark_dec = None

        x_dec = torch.zeros(size=(len(insample_y), self.h, 1), device=insample_y.device)
        x_dec = torch.cat([insample_y[:, -self.label_len :, :], x_dec], dim=1)

        enc_out = self.enc_embedding(insample_y, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # attns visualization

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        forecast = self.loss.domain_map(dec_out[:, -self.h :])
        return forecast
