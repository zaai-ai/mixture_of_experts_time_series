from pytorch_lightning.callbacks import Callback
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.patches as mpatches
from tsfeatures import tsfeatures
from functools import partial
import warnings

# Suppress warnings from tsfeatures
warnings.filterwarnings("ignore", category=FutureWarning)

class SeriesSimilarityCallback(Callback):
    def __init__(self, *, training_df: pd.DataFrame):
        """
        Args:
            training_df: DataFrame with training data containing 'unique_id' and 'y'.
        """
        super().__init__()
        self.training_df = training_df
        self.expert_colors = None  
        self.counter = 0


    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            series_ids = self.training_df['unique_id'].unique()
            tsfeatures_list = []  # Store extracted time series features
            series_labels = []
            expert_indices = []
            gate_probs_list = []  

            print(f"Computing series embedding for {len(series_ids)} series...")

            # Calculate features for all series in the training DataFrame
            ts_feats = tsfeatures(self.training_df, freq=12)


            for series_id in series_ids:
                
                df_series = self.training_df[self.training_df['unique_id'] == series_id]

                tsfeatures_np = ts_feats[ts_feats['unique_id'] == series_id].drop(columns=["unique_id"]).iloc[0].to_numpy()  # Convert to NumPy array

                tsfeatures_np = np.nan_to_num(tsfeatures_np, nan=0.0)
                tsfeatures_list.append(tsfeatures_np)
                series_labels.append(series_id)

                inputs = df_series['y'].values[-24:]

                # Normalize the inputs using the training statistics.
                # inputs = (inputs - np.mean(inputs)) / np.std(inputs)

                # Create an input tensor of shape (1, 24) and move it to the model's device.
                input_tensor = torch.tensor(inputs, device=pl_module.device).float().unsqueeze(0)

                # Pass the tensor through the gate network.
                gate_logits = pl_module.gate(input_tensor)  # Expected shape: (1, n_experts)
                # Convert logits to probabilities using the module's softmax.
                gate_probs_tensor = pl_module.softmax(gate_logits)
                # Detach, move to CPU, and squeeze to get a 1D array of shape (n_experts,).
                gate_probs_np = gate_probs_tensor.detach().cpu().numpy().squeeze()
                gate_probs_list.append(gate_probs_np)

                expert_indices.append(np.argmax(gate_probs_np))

            print(f"Extracted time series features for {len(tsfeatures_list)} series.")

            if len(tsfeatures_list) == 0:
                return

            tsfeatures_array = np.array(tsfeatures_list)
            gate_probs = np.array(gate_probs_list)
            n_series, n_experts = gate_probs.shape


            # Initialize a fixed expert-to-color mapping on the first call.
            if self.expert_colors is None:
                # Use 'tab10' if n_experts <= 10, otherwise fall back to 'tab20'
                cmap = plt.get_cmap('tab10') if n_experts <= 10 else plt.get_cmap('tab20')
                self.expert_colors = np.array([cmap(i) for i in range(n_experts)])


            # Perform t-SNE on tsfeatures
            tsne = TSNE(n_components=2, random_state=42)
            print("Computing t-SNE embedding...")
            embedding_2d = tsne.fit_transform(tsfeatures_array)
            print("t-SNE embedding computed.")
            # Assign colors
            series_colors = np.array([self.expert_colors[idx] for idx in expert_indices])

            try:
                # Create scatter plot
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(
                    embedding_2d[:, 0],
                    embedding_2d[:, 1],
                    c=series_colors,  
                    s=50,
                    alpha=0.7
                )

                for i, label in enumerate(series_labels):
                    ax.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]),
                                fontsize=5, alpha=0.6)

                ax.set_title('2D Embedding of Time Series Features')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')

                plt.tight_layout()
                plt.show()
                self.counter += 1
            except Exception as e:
                print("An error occurred while plotting the t-SNE embedding:", e)

        except Exception as e:
            print("An error occurred while computing the series embedding:", e)
