from pytorch_lightning.callbacks import Callback
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # For nonlinear 2D embedding
import numpy as np
import matplotlib.patches as mpatches  # For legend patches

class SeriesDistributionCallback(Callback):
    def __init__(self, *, training_df: pd.DataFrame):
        """
        Args:
            training_df: DataFrame with training data containing 'unique_id' and 'y'.
        """
        super().__init__()
        self.training_df = training_df
        self.expert_colors = None  # Will be initialized once we know the number of experts
        self.counter=0

    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            # Get unique series identifiers from the DataFrame.
            series_ids = self.training_df['unique_id'].unique()
            gate_probs_list = []  # To store the gate probability vectors.
            series_labels = []    # To keep track of which series each vector belongs to.
            expert_indices = []   # To store the most important expert for each series.

            # Iterate over each series.
            for series_id in series_ids:
                df_series = self.training_df[self.training_df['unique_id'] == series_id]
                # Ensure the series has enough data (e.g., at least the last 24 observations).
                if len(df_series) < 24:
                    continue

                # Extract the last 24 values.
                inputs = df_series['y'].values[-24:]
                # Create an input tensor of shape (1, 24) and move it to the model's device.
                input_tensor = torch.tensor(inputs, device=pl_module.device).float().unsqueeze(0)

                # Pass the tensor through the gate network.
                gate_logits = pl_module.gate(input_tensor)  # Expected shape: (1, n_experts)
                # Convert logits to probabilities using the module's softmax.
                gate_probs_tensor = pl_module.softmax(gate_logits)
                # Detach, move to CPU, and squeeze to get a 1D array of shape (n_experts,).
                gate_probs_np = gate_probs_tensor.detach().cpu().numpy().squeeze()

                gate_probs_list.append(gate_probs_np)
                series_labels.append(series_id)
                expert_indices.append(np.argmax(gate_probs_np))  # Store the most important expert index

            # If no series were processed, exit early.
            if len(gate_probs_list) == 0:
                return

            # Convert list of gate probability vectors to a numpy array of shape (n_series, n_experts).
            gate_probs = np.array(gate_probs_list)
            n_series, n_experts = gate_probs.shape

            # Initialize a fixed expert-to-color mapping on the first call.
            if self.expert_colors is None:
                # Use 'tab10' if n_experts <= 10, otherwise fall back to 'tab20'
                cmap = plt.get_cmap('tab10') if n_experts <= 10 else plt.get_cmap('tab20')
                self.expert_colors = np.array([cmap(i) for i in range(n_experts)])

            # Assign each series the color of its most important expert.
            series_colors = np.array([self.expert_colors[idx] for idx in expert_indices])

            # Perform dimensionality reduction to 2D using t-SNE.
            tsne = TSNE(n_components=2, random_state=42)
            embedding_2d = tsne.fit_transform(gate_probs)

            # Create a scatter plot of the 2D embeddings using the computed colors.
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

            ax.set_title('2D Embedding of Series Gate Distributions')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')

            # Add a legend on the right showing expert colors
            expert_patches = [
                mpatches.Patch(color=self.expert_colors[i], label=f'Expert {i}')
                for i in range(n_experts)
            ]
            ax.legend(handles=expert_patches, title="Experts", loc="upper right", bbox_to_anchor=(1.2, 1))

            plt.tight_layout()
            plt.savefig('./callback_images/' + str(self.counter) + '.png')
            plt.show()
            self.counter += 1
            # plt.close(fig)

        except Exception as e:
            print("An error occurred while computing the series embedding:", e)
