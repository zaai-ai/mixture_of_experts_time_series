import pandas as pd
from datasetsforecast.m3 import M3
from datasetsforecast.m4 import M4

# Define directories for saving datasets
m3_dir = './data/m3'
m4_dir = './data/m4'

# Define groups for M3 and M4
m3_groups = ['Yearly', 'Quarterly', 'Monthly', 'Other']
m4_groups = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']

def download_m3_data():
    """Download all M3 Competition datasets and combine them."""
    print("Downloading M3 datasets...")
    m3_all_series = []
    for group in m3_groups:
        print(f"Downloading M3 group: {group}")
        m3_group_data = M3.load(directory=m3_dir, group=group)
        m3_all_series.append(m3_group_data[0])
    m3_all = pd.concat(m3_all_series, ignore_index=True)
    print(f"Total series in M3: {m3_all['unique_id'].nunique()}")
    return m3_all

def download_m4_data():
    """Download all M4 Competition datasets and combine them."""
    print("Downloading M4 datasets...")
    m4_all_series = []
    for group in m4_groups:
        print(f"Downloading M4 group: {group}")
        m4_group_data = M4.load(directory=m4_dir, group=group)
        m4_all_series.append(m4_group_data[0])
    m4_all = pd.concat(m4_all_series, ignore_index=True)
    print(f"Total series in M4: {m4_all['unique_id'].nunique()}")
    return m4_all

def save_to_csv(dataframe, filename):
    """Save a dataframe to a CSV file."""
    dataframe.to_csv(filename, index=False)
    print(f"Saved dataset to {filename}")

def main():
    """Main function to download, combine, and save M3 and M4 datasets."""
    # Download datasets
    m3_all = download_m3_data()
    m4_all = download_m4_data()

    # Combine M3 and M4 datasets
    combined = pd.concat([m3_all, m4_all], ignore_index=True)
    print(f"Total series across M3 and M4: {combined['unique_id'].nunique()}")


if __name__ == "__main__":
    main()
