import numpy as np 
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from tsfeatures import tsfeatures  # Requires tsfeatures package
import pandas as pd


class GateValuesCollectorCallback(pl.Callback):
    """
    A PyTorch Lightning callback to collect gate values per epoch
    and analyze expert specialization in a mixture-of-experts model.
    """
    def __init__(self, top_k: int = 2, nr_layers: int = 3) -> None:
        super().__init__()
        self._gate_values = []  # Stores gate values for all layers
        self.top_k = top_k
        self.nr_layers = nr_layers  # Number of layers to track

    def on_predict_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each epoch during training.
        Collects and processes gate values from all layers.
        """
        all_gate_values = pl_module.all_gate_logits  # Assuming this exists
        all_inputs = pl_module.all_inputs  # Assuming this exists

        if not all_gate_values or len(all_gate_values[0]) < self.nr_layers:
            print("Not enough layers in all_gate_logits.")
            return

        epoch_gate_values = []  # Collect values per layer
        epoch_inputs = []  # Collect inputs per layer
        for layer_idx in range(self.nr_layers):
            gate_values = all_gate_values[layer_idx]  # Extract for this layer

            # Concatenate gate values across all batches in the list
            gate_concated_values = torch.cat(gate_values, dim=0)  # (all_test_points, num_experts)
            all_concated_inputs = torch.cat(all_inputs[layer_idx], dim=0)  # (all_test_points, num_features)

            # Extract top-k experts per sample
            top_k_gate_values, topk_indices = torch.topk(gate_concated_values, self.top_k, dim=1)
            layer_gate_values = torch.zeros_like(gate_concated_values)
            top_k_probs = torch.softmax(top_k_gate_values, dim=1)
            layer_gate_values.scatter_(1, topk_indices, top_k_probs)

            epoch_gate_values.append(layer_gate_values.detach().cpu().numpy())
            epoch_inputs.append(all_concated_inputs.detach().cpu().numpy())
        
        self._gate_values.append(np.array(epoch_gate_values))  # Store per epoch
        self.plot_expert_density()

        ## save the gate values to a file
        np.save(f"gate_values_epoch_{trainer.current_epoch}.npy", np.array(epoch_gate_values))
        np.save(f"all_inputs_epoch_{trainer.current_epoch}.npy", np.array(epoch_inputs))

    def plot_expert_density(self):
        """
        Plots a heatmap of expert selection probability per layer.
        """
        if not self._gate_values:
            print("No data collected yet.")
            return

        # Aggregate across epochs and batch
        all_values = np.mean(np.concatenate(self._gate_values, axis=1), axis=1)  # (layers, experts)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(all_values, aspect='auto', cmap='Blues', interpolation='none')
        plt.colorbar(label="Average Probability")
        plt.xlabel("Experts")
        plt.ylabel("Layers")
        plt.title("Gating Scores for Experts Across Layers")
        plt.xticks(np.arange(all_values.shape[1]))
        plt.yticks(np.arange(all_values.shape[0]))
        plt.show()

    def analyze_expert_specialization(self, time_series_data: pd.DataFrame):
        """
        Analyzes expert specialization based on time series characteristics.
        """
        if not self._gate_values:
            print("No data collected yet.")
            return

        # Compute TS Features
        features = tsfeatures(time_series_data["series"])
        features = pd.DataFrame(features)

        # Get average gate values per series
        avg_gate_values = np.mean(self._gate_values[-1], axis=1)  # Avg over batch

        # Correlate with time series features
        correlation = pd.DataFrame(avg_gate_values).corrwith(features)

        print("Correlation between expert selection and time series features:")
        print(correlation)

        # Scatter plot: Trend vs Expert Selection
        plt.figure(figsize=(6, 4))
        plt.scatter(features["trend"], avg_gate_values, alpha=0.6)
        plt.xlabel("Trend Strength")
        plt.ylabel("Expert Selection Probability")
        plt.title("Expert Specialization vs Trend")
        plt.show()
