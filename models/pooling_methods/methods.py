import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

__ALL__ = ["BasePooling", "DensePooling", "SparsePooling", "SoftPooling", "SharedExpertPooling"]

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
        device: Optional[torch.device] = None,
        unpack: bool = True,
        return_soft_gates: bool = False
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
        self.unpack: bool = unpack
        self.return_soft_gates: bool = return_soft_gates
        
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
    def forward(self, windows_batch: dict) -> torch.Tensor:

        if self.unpack: insample_y = windows_batch['insample_y']
        else: insample_y = windows_batch

        # Compute the gate and normalize it.
        gate_logits: torch.Tensor = self.gate(insample_y)
        gate_probs: torch.Tensor = self.softmax(gate_logits)
        
        # Initialize the weighted sum.
        weighted_sum: torch.Tensor = torch.zeros(
            insample_y.size(0), self.out_features, device=insample_y.device
        )
        # Sum over all experts.
        for i, expert in enumerate(self.experts):
            weighted_sum += gate_probs[:, i].unsqueeze(1) * expert(windows_batch)

        if self.return_soft_gates:
            return weighted_sum, gate_probs

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
        device: Optional[torch.device] = None,
        unpack: bool = True,
        return_soft_gates: bool = False,
        bias: bool = False
    ) -> None:
        """
        Args:
            experts (List[nn.Module]): List of expert models.
            gate (nn.Module): Gating network that computes the weights.
            out_features (int): The number of output features.
            k (int, optional): The number of top experts to select. Defaults to 1.
            device (Optional[torch.device], optional): Device to run on. Defaults to CPU.
        """
        super(SparsePooling, self).__init__(experts, gate, out_features, device, unpack, return_soft_gates)
        self.k: int = k

        self.bias: nn.Parameter = nn.Parameter(torch.empty(len(experts))) if bias else None

    def forward(self, windows_batch: dict) -> torch.Tensor:

        if self.unpack: insample_y = windows_batch['insample_y']
        else: insample_y = windows_batch

        # Compute the gate logits. Shape: [batch, num_experts]
        gate_logits: torch.Tensor = self.gate(insample_y)

        # original logits
        original_gate_logits = gate_logits.clone()

        if self.bias is not None:
            gate_logits += self.bias

        # Select the top-k experts for each sample.
        # topk_values & topk_indices have shape: [batch, k]
        topk_values, topk_indices = torch.topk(gate_logits, self.k, dim=1)

        # print(f"topk_indices: {topk_indices}")

        if self.bias is not None:
            topk_original_values = torch.gather(original_gate_logits, 1, topk_indices)
        else:
            topk_original_values = topk_values

        # Compute probabilities for the top-k experts using softmax.
        gate_probs: torch.Tensor = self.softmax(topk_original_values)

        # Initialize the weighted sum output.
        weighted_sum = torch.zeros(
            insample_y.size(0), self.out_features, device=insample_y.device
        )
        
        num_experts = len(self.experts)
        # Group contributions by expert.
        for expert_idx in range(num_experts):

            # Create a mask of shape [batch, k] indicating where expert_idx was selected.
            expert_mask = (topk_indices == expert_idx)
            if expert_mask.sum() == 0:
                continue  # This expert was not selected in any top-k.

            # Sum the corresponding probabilities for each sample.
            # This gives a weight for each sample for expert_idx.
            expert_weight = (gate_probs * expert_mask.float()).sum(dim=1)  # Shape: [batch]

            # Compute expert output for the entire batch.
            expert_output = self.experts[expert_idx](windows_batch)  # Shape: [batch, out_features]

            # Add the weighted expert output.
            weighted_sum += expert_output * expert_weight.unsqueeze(1)

        if self.return_soft_gates:
            return weighted_sum, gate_logits

        return weighted_sum

################################################################################
# Shared Expert Pooling
################################################################################
class SharedExpertPooling(BasePooling):
    """
    Uses one shared expert and sparse pooling for the others
    """
    def __init__(
        self,
        experts: List[nn.Module],
        shared_expert: nn.Module,
        gate: nn.Module,
        sparse_gate: nn.Module,
        out_features: int,
        k: int = 3,
        device: Optional[torch.device] = None,
        unpack: bool = True,
        return_soft_gates: bool = False,
        bias: bool = False
    ) -> None:
        """
        Args:
            experts (List[nn.Module]): List of expert models.
            gate (nn.Module): Gating network that computes the weights.
            out_features (int): The number of output features.
            k (int, optional): The number of top experts to select. Defaults to 1.
            device (Optional[torch.device], optional): Device to run on. Defaults to CPU.
        """
        super(SharedExpertPooling, self).__init__(experts, gate, out_features, device, unpack, return_soft_gates)
        self.k: int = k

        self.bias: nn.Parameter = nn.Parameter(torch.empty(len(experts))) if bias else None

        self.shared_expert: nn.Module = shared_expert
        self.sparse_pooling = SparsePooling(
            experts=experts,
            gate=sparse_gate,
            out_features=out_features,
            k=k,
            unpack=unpack,
            return_soft_gates=return_soft_gates,
            bias=bias
        )

    def forward(self, windows_batch: dict) -> torch.Tensor:
        
        if self.unpack: insample_y = windows_batch['insample_y']
        else: insample_y = windows_batch

        gate_logits: torch.Tensor = self.gate(insample_y)

        gate_probs: torch.Tensor = self.softmax(gate_logits)


        # Initialize the weighted sum output.
        weighted_sum = torch.zeros(
            insample_y.size(0), self.out_features, device=insample_y.device
        )

        # Compute the shared expert output.
        shared_expert_output = self.shared_expert(windows_batch)
        # Add the shared expert output to the weighted sum.
        weighted_sum += shared_expert_output * gate_probs[:, 0].unsqueeze(1)
        
        # Compute the sparse pooling output.
        sparse_pooling_output, sparse_gate_probs = self.sparse_pooling(windows_batch)
        # Add the sparse pooling output to the weighted sum.
        weighted_sum += sparse_pooling_output * gate_probs[:, 1].unsqueeze(1)

        return weighted_sum, sparse_gate_probs


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
        device: Optional[torch.device] = None,
        unpack: bool = True,
        return_soft_gates: bool = False
    ) -> None:
        """
        Args:
            experts (List[nn.Module]): List of expert models.
            gate (nn.Module): Gating network that computes the weights.
            out_features (int): The number of output features.
            temperature (float, optional): Temperature parameter for softmax. Defaults to 1.0.
            device (Optional[torch.device], optional): Device to run on. Defaults to CPU.
            unpack (bool, optional): Whether to unpack the input dictionary. Defaults to True.
        """
        super().__init__(experts, gate, out_features, device, unpack)
        self.temperature: float = temperature

    def forward(self, windows_batch: dict) -> torch.Tensor:

        if self.unpack: insample_y = windows_batch['insample_y']
        else: insample_y = windows_batch

        # Compute the gate logits and apply temperature-scaled softmax.
        gate_logits: torch.Tensor = self.gate(insample_y)
        soft_gate: torch.Tensor = F.softmax(gate_logits / self.temperature, dim=1)
        
        weighted_sum: torch.Tensor = torch.zeros(
            insample_y.size(0), self.out_features, device=insample_y.device
        )
        # Sum over all experts.
        for i, expert in enumerate(self.experts):
            weighted_sum += soft_gate[:, i].unsqueeze(1) * expert(windows_batch)

        if self.return_soft_gates:
            return weighted_sum, soft_gate

        return weighted_sum
