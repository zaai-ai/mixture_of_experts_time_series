from pytorch_lightning.callbacks import Callback

import pandas as pd
import torch
import matplotlib.pyplot as plt

class GateDistributionCallback(Callback):

    def __init__(self,
                 *,
                 training_df: pd.DataFrame):
        super().__init__()

        self.training_df = training_df

    def on_validation_epoch_end(self, trainer, pl_module):
        
        ## plot the distributions of the gates selected by the model for each
        ## for each of the series get the expert that should take care of it
        ## and plot the distribution of the gates for each of the series

        # get the gate distribution
        try:
            df_unique_id_999 = self.training_df[self.training_df['unique_id'] == "M999"]

            # the input tensor should be of shape (1, gate.input_size)
            inputs_id = df_unique_id_999['y'].values[-24:]
            input_tensor = torch.tensor(inputs_id, device=pl_module.device).float().unsqueeze(0)
            

            gate_dist = pl_module.gate(input_tensor)
            gate_2 = gate_dist.detach().cpu().numpy()
            gate_dist = pl_module.softmax(gate_dist)
            gate_dist = gate_dist.detach().cpu().numpy()
            print(gate_dist)
            print(gate_2)
            gate_dist = gate_dist.reshape(-1, 1)

            gate_df = pd.DataFrame(gate_dist, columns=['gate'])
            # plot the distribution of the gates
            # Plot the gate distribution as a bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(gate_df.index, gate_df['gate'], color='skyblue')
            ax.set_xlabel('Expert Index')
            ax.set_ylabel('Gate Probability')
            ax.set_title('Gate Distribution for Series M999')
            plt.tight_layout()

            # Display the plot (or alternatively, you could save the figure)
            plt.show()
            plt.close(fig)

            gate_2 = gate_2.reshape(-1, 1)
            gate_2_df = pd.DataFrame(gate_2, columns=['gate'])
            ## plot the values of the gate_2
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(gate_2_df)
            ax.set_xlabel('Expert Index')
            ax.set_ylabel('Gate Probability')
            ax.set_title('Gate Distribution for Series M999')
            plt.tight_layout()

            # Display the plot (or alternatively, you could save the figure)
            plt.show()
            plt.close(fig)

        except Exception as e:
            print(e)
            pass
       
        

