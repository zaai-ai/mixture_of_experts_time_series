import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

__ALL__ = ["BasePooling", "DensePooling", "SparsePooling", "SoftPooling"]

###############################################################################
# Base Pooling Module
###############################################################################
class BasePooling(nn.Module):
    """
    Base class for pooling strategies in a Mixture-of-Experts setting.
    It holds a list of expert modules and a gating network.
    """
    def __init__(
        self,
        experts: List[nn.Module],
        gate: nn.Module,
        out_features: int,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Args:
            experts (List[nn.Module]): List of expert models.
            gate (nn.Module): Gating network that computes the weights.
            out_features (int): The number of output features (each expert’s output size).
            device (Optional[torch.device], optional): Device to run on. Defaults to CPU.
        """
        super(BasePooling, self).__init__()
        self.experts: nn.ModuleList = nn.ModuleList(experts)
        self.gate: nn.Module = gate
        self.out_features: int = out_features
        self.device: torch.device = device if device is not None else torch.device("cpu")
        self.softmax: nn.Softmax = nn.Softmax(dim=1)
        
    def forward(self, insample_y: torch.Tensor) -> torch.Tensor:
        """
        Forward method to be implemented by subclasses.
        
        Args:
            insample_y (torch.Tensor): Input tensor with shape (batch_size, input_size).
        
        Returns:
            torch.Tensor: The weighted sum of experts' outputs.
        """
        raise NotImplementedError("Subclasses should implement this method.")

###############################################################################
# Dense Pooling
###############################################################################
class DensePooling(BasePooling):
    """
    Dense pooling uses all experts. The gate is computed and softmax-normalized
    (similar to your provided snippet), then each expert’s output is weighted
    accordingly and summed.
    """
    def forward(self, insample_y: torch.Tensor) -> torch.Tensor:
        # Compute the gate and normalize it.
        gate_logits: torch.Tensor = self.gate(insample_y)
        gate_probs: torch.Tensor = self.softmax(gate_logits)
        
        # Initialize the weighted sum.
        weighted_sum: torch.Tensor = torch.zeros(
            insample_y.size(0), self.out_features, device=insample_y.device
        )
        # Sum over all experts.
        for i, expert in enumerate(self.experts):
            weighted_sum += gate_probs[:, i].unsqueeze(1) * expert(insample_y)
        return weighted_sum

###############################################################################
# Sparse Pooling
###############################################################################
class SparsePooling(BasePooling):
    """
    Sparse pooling uses only the top-k experts (as determined by the gating network).
    """
    def __init__(
        self,
        experts: List[nn.Module],
        gate: nn.Module,
        out_features: int,
        k: int = 3,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Args:
            experts (List[nn.Module]): List of expert models.
            gate (nn.Module): Gating network that computes the weights.
            out_features (int): The number of output features.
            k (int, optional): The number of top experts to select. Defaults to 1.
            device (Optional[torch.device], optional): Device to run on. Defaults to CPU.
        """
        super(SparsePooling, self).__init__(experts, gate, out_features, device)
        self.k: int = k

    def forward(self, insample_y: torch.Tensor) -> torch.Tensor:
        # Compute the gate logits. Shape: [batch, num_experts]
        gate_logits: torch.Tensor = self.gate(insample_y)

        # Select the top-k experts for each sample.
        # topk_values & topk_indices have shape: [batch, k]
        topk_values, topk_indices = torch.topk(gate_logits, self.k, dim=1)

        # Compute probabilities for the top-k experts using softmax.
        gate_probs: torch.Tensor = self.softmax(topk_values)

        # Initialize the weighted sum output.
        weighted_sum = torch.zeros(
            insample_y.size(0), self.out_features, device=insample_y.device
        )
        
        num_experts = len(self.experts)
        # Group contributions by expert.
        # TODO: check if it is working
        for expert_idx in range(num_experts):

            # Create a mask of shape [batch, k] indicating where expert_idx was selected.
            expert_mask = (topk_indices == expert_idx)
            if expert_mask.sum() == 0:
                continue  # This expert was not selected in any top-k.

            # Sum the corresponding probabilities for each sample.
            # This gives a weight for each sample for expert_idx.
            expert_weight = (gate_probs * expert_mask.float()).sum(dim=1)  # Shape: [batch]

            # Compute expert output for the entire batch.
            expert_output = self.experts[expert_idx](insample_y)  # Shape: [batch, out_features]

            # Add the weighted expert output.
            weighted_sum += expert_output * expert_weight.unsqueeze(1)

        return weighted_sum

###############################################################################
# Soft Pooling
###############################################################################
#### TODO: DO THIS ONE
class SoftPooling(BasePooling):
    """
    Soft pooling applies a temperature-scaled softmax to the gate before weighting
    and summing the experts' outputs.
    """
    def __init__(
        self,
        experts: List[nn.Module],
        gate: nn.Module,
        out_features: int,
        temperature: float = 1.0,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Args:
            experts (List[nn.Module]): List of expert models.
            gate (nn.Module): Gating network that computes the weights.
            out_features (int): The number of output features.
            temperature (float, optional): Temperature parameter for softmax. Defaults to 1.0.
            device (Optional[torch.device], optional): Device to run on. Defaults to CPU.
        """
        super(SoftPooling, self).__init__(experts, gate, out_features, device)
        self.temperature: float = temperature

    def forward(self, insample_y: torch.Tensor) -> torch.Tensor:
        # Compute the gate logits and apply temperature-scaled softmax.
        gate_logits: torch.Tensor = self.gate(insample_y)
        soft_gate: torch.Tensor = F.softmax(gate_logits / self.temperature, dim=1)
        
        weighted_sum: torch.Tensor = torch.zeros(
            insample_y.size(0), self.out_features, device=insample_y.device
        )
        # Sum over all experts.
        for i, expert in enumerate(self.experts):
            weighted_sum += soft_gate[:, i].unsqueeze(1) * expert(insample_y)
        return weighted_sum
