import numpy as np 
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from tsfeatures import tsfeatures
import pandas as pd


class GateValuesCollectorCallback(pl.Callback):
    """
    A PyTorch Lightning callback to collect gate values per epoch
    and analyze expert specialization in a mixture-of-experts model.
    """
    def __init__(self, top_k: int = 2, nr_layers: int = 3, is_stack: bool =False, reset_on_epoch: bool = False) -> None:
        super().__init__()
        self._gate_values = []  # Stores gate values for all layers
        self.top_k = top_k
        self.nr_layers = nr_layers  # Number of layers to track
        self.is_stack = is_stack  # Flag to indicate if the model is a stackMoe
        self.reset_on_epoch = reset_on_epoch  # Flag to reset gate values on each epoch
        self.batch = 0

    def on_predict_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each epoch during training.
        Collects and processes gate values from all layers.
        """
        all_gate_values = pl_module.all_gate_logits  # Assuming this exists
        all_inputs = pl_module.all_inputs  # Assuming this exists
        # all_outputs = pl_module.all_outs  # Assuming this exists

        if not all_gate_values or len(all_gate_values[0]) < self.nr_layers:
            print("Not enough layers in all_gate_logits.")
            return

        epoch_gate_values = []  # Collect values per layer
        epoch_inputs = []  # Collect inputs per layer
        
        if self.is_stack:
            all_gates_cat = torch.cat(all_gate_values, dim=0)
            all_inputs_cat = torch.cat(all_inputs, dim=0)
            # all_outputs = torch.cat([torch.cat(batch, dim=0) for batch in all_outputs], dim=0)
            print(f"\nall_gates_cat shape: {all_gates_cat.shape}")
            print("\nmean inputs_cat: ", all_gates_cat.mean(dim=0))

            # count the number of experts > 0.4
            num_experts = (all_gates_cat > 0.5).sum(dim=0)
            print(f"\nnum_experts: {num_experts}")

            # Count the number of times each expert has the highest gate value
            best_expert_counts = torch.argmax(all_gates_cat, dim=1).bincount(minlength=all_gates_cat.shape[1])
            print(f"\nBest expert counts: {best_expert_counts}")

            self.plot_on_stack_analysis(all_gates_cat, all_inputs_cat)

            np.save(f"gate_values_stack{self.batch}_epoch_{trainer.current_epoch}.npy", np.array(all_gates_cat.detach().cpu().numpy()))
            np.save(f"all_inputs_stack{self.batch}_epoch_{trainer.current_epoch}.npy", np.array(all_inputs_cat.detach().cpu().numpy()))
            # np.save(f"all_outputs_stack{self.batch}_epoch_{trainer.current_epoch}.npy", np.array(all_outputs.detach().cpu().numpy()))

            self.batch += 1
            
        else:
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

        if self.reset_on_epoch:
            print("Resetting gate values for next epoch.")
            pl_module.all_gate_logits = []
            pl_module.all_inputs = []


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
        
    def plot_on_stack_analysis(self, gate_values, inputs):
        
        ### select 10 most samples with highest gate values per dimension/expert
        top_k = 10  # number of top values to select
        top_k_values, topk_indices = torch.topk(gate_values, k=top_k, dim=0)

        # print(f"topk_indices shape: {topk_indices.shape}")
        # print(f"topk_indices: {topk_indices}")
        # print(f"topk_values shape: {top_k_values.shape}")
        # print(f"topk_values: {top_k_values}")
        # topk_indices: shape [top_k, num_experts]

        for expert_idx in range(topk_indices.shape[1]):
            expert_top_inputs = inputs[topk_indices[:, expert_idx]]
            # print(inputs[topk_indices[:, expert_idx][0]])
            # print(topk_indices[:, expert_idx])
            # print(expert_top_inputs)

            plt.figure(figsize=(10, 6))
            for i in range(expert_top_inputs.shape[0]):
                # print(f"expert_top_inputs[:, {i}]: {expert_top_inputs[i]}")
                plt.plot(expert_top_inputs[i].detach().cpu().numpy(), label=f"Input {i+1}")
                # plt.plot(expert_top_inputs[0:10], color='blue', label="Identity")
            # plt.plot(expert_top_inputs[1], color='red', label="Trend")
            # plt.plot(expert_top_inputs[2], color='green', label="Seasonality")
            plt.title(f"Top 3 Inputs for Expert {expert_idx} (I->T->S)")
            plt.xlabel("Lags")
            plt.ylabel("Input Value")
            plt.legend()
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
