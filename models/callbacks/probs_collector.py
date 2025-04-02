from typing import List, Any
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


class GateValuesCollectorCallback(Callback):
    """
    A PyTorch Lightning callback to collect gate values from model outputs
    during prediction.
    """
    def __init__(self, top_k : int = 2, layer_to_check: int = 0) -> None:
        super().__init__()
        self._gate_values: List[np.ndarray] = []

        self.top_k = top_k
        self.layer_to_check = layer_to_check

    def on_predict_batch_end(self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,) -> None:
        """
        Called when a batch ends during prediction. Collects gate values if they exist in the model's output.
        
        Args:
            outputs (Any): Model outputs from the batch.
            batch (Any): Input batch data (not used in this callback but required by Lightning API).
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader (default: 0).
        """
        all_gate_values = pl_module.all_gate_logits

        print(f"length of all_gate_values: {len(all_gate_values)}")
        print(f"all_gate_values: {all_gate_values[0]}")
        print(f"all_gate_values: {len(all_gate_values[0])}")


        all_gate_values = all_gate_values[0][self.layer_to_check]

        top_k_gate_values, topk_indices = torch.topk(all_gate_values, self.top_k, dim=1)

        # put all values in the list and set the top k probs to selected values
        all_gate_values = torch.zeros_like(all_gate_values)

        top_k_probs = torch.softmax(top_k_gate_values, dim=1)

        all_gate_values.scatter_(1, topk_indices, top_k_probs)

        all_gate_values = all_gate_values.detach().cpu().numpy()

        self._gate_values.append(all_gate_values)

        print(f"top_k_gate_values: {all_gate_values}")


    def get_collected_gate_values(self) -> List[np.ndarray]:
        """
        Returns the collected gate values.
        
        Returns:
            List[np.ndarray]: A list of collected gate values as NumPy arrays.
        """
        return self._gate_values