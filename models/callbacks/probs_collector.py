from typing import List, Any
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


class GateValuesCollectorCallback(Callback):
    """
    A PyTorch Lightning callback to collect gate values from model outputs
    during prediction.
    """
    def __init__(self, top_k : int = 1) -> None:
        super().__init__()
        self._gate_values: List[np.ndarray] = []
        self.top_k = top_k

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
        print(f"outputs: {outputs}")
        # if isinstance(outputs, dict) and 'gate_values' in outputs:
        #     gate_values = outputs['gate_values']
        #     if isinstance(gate_values, torch.Tensor):
        #         self._gate_values.append(gate_values.detach().cpu().numpy())

    def get_collected_gate_values(self) -> List[np.ndarray]:
        """
        Returns the collected gate values.
        
        Returns:
            List[np.ndarray]: A list of collected gate values as NumPy arrays.
        """
        return self._gate_values