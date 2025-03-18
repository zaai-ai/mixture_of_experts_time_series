
import pandas as pd
from datasetsforecast.m3 import M3  
from datasetsforecast.m4 import M4
from omegaconf import DictConfig

def load_dataset(dataset_name: str, dataset_cfg: DictConfig):
    """Load dataset based on dataset_name and its configuration."""
    if dataset_name == "m3":
        print("Loading m3_monthly dataset...")
        return M3.load(
            directory=dataset_cfg.directory,
            group=dataset_cfg.group)[0]
    elif dataset_name == "m4":
        print("Loading m4_monthly dataset...")
        df = M4.load(
            directory=dataset_cfg.directory,
            group=dataset_cfg.group)[0]
        
        # Convert the 'ds' to integer
        df['ds'] = pd.to_datetime(df['ds']).astype(int)

        return df
    else:
        raise ValueError(
            f"Loading method for dataset '{dataset_name}' is not defined.")

def train_test_split(df: pd.DataFrame, horizon: int):
    """Split the dataframe into training and test sets by horizon."""
    groups = df.groupby('unique_id')
    train_list, test_list = [], []
    for _, group_df in groups:
        group_df = group_df.sort_values('ds')
        train_list.append(group_df.head(-horizon))
        test_list.append(group_df.tail(horizon))
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df, test_df