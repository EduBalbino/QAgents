import pandas as pd
import numpy as np
import os

# Define the paths to the datasets
DATASET_DIR = 'Datasets'
CICIDS_PATH = os.path.join(DATASET_DIR, 'CICIDS2017.csv')
UNSW_PATH = os.path.join(DATASET_DIR, 'UNSW_NB15.csv')
ML_EDGE_PATH = os.path.join(DATASET_DIR, 'ML-EdgeIIoT-dataset.csv')
OUTPUT_PATH = 'unified_dataset_full.csv'

# Define the column mappings from original names to unified names
CICIDS_MAP = {
    'Src IP': 'src_ip',
    'Dst IP': 'dst_ip',
    'Dst Port': 'dst_port',
    'Protocol': 'protocol',
    'Flow Duration': 'duration',
    'Tot Fwd Pkts': 'fwd_packets',
    'Tot Bwd Pkts': 'bwd_packets',
    'TotLen Fwd Pkts': 'fwd_bytes',
    'TotLen Bwd Pkts': 'bwd_bytes',
    'Fwd Pkt Len Mean': 'fwd_pkt_len_mean',
    'Bwd Pkt Len Mean': 'bwd_pkt_len_mean',
    'Fwd Pkt Len Std': 'fwd_pkt_len_std',
    'Bwd Pkt Len Std': 'bwd_pkt_len_std',
    'Fwd IAT Mean': 'fwd_iat_mean',
    'Bwd IAT Mean': 'bwd_iat_mean',
    'Fwd IAT Std': 'fwd_iat_std',
    'Bwd IAT Std': 'bwd_iat_std',
    'Fwd IAT Tot': 'fwd_iat_total',
    'Bwd IAT Tot': 'bwd_iat_total',
    'Fwd Win Byts': 'fwd_win_bytes',
    'Bwd Win Byts': 'bwd_win_bytes',
    'Label': 'attack_category'
}

UNSW_MAP = {
    'proto': 'protocol',
    'dur': 'duration',
    'spkts': 'fwd_packets',
    'dpkts': 'bwd_packets',
    'sbytes': 'fwd_bytes',
    'dbytes': 'bwd_bytes',
    'smean': 'fwd_pkt_len_mean',
    'dmean': 'bwd_pkt_len_mean',
    'sinpkt': 'fwd_iat_mean',
    'dinpkt': 'bwd_iat_mean',
    'sjit': 'fwd_iat_std',
    'djit': 'bwd_iat_std',
    'swin': 'fwd_win_bytes',
    'dwin': 'bwd_win_bytes',
    'tcprtt': 'tcp_rtt',
    'synack': 'synack_time',
    'ackdat': 'ackdat_time',
    'attack_cat': 'attack_category',
    'label': 'is_attack'
}

ML_EDGE_MAP = {
    'ip.src_host': 'src_ip',
    'ip.dst_host': 'dst_ip',
    'tcp.dstport': 'dst_port',
    'tcp.srcport': 'src_port',
    'Attack_type': 'attack_category',
    'Attack_label': 'is_attack'
}

def preprocess_cicids(df):
    """Preprocesses the CICIDS2017 dataset."""
    # Clean column names by stripping leading/trailing whitespace
    df.columns = df.columns.str.strip()

    df.rename(columns=CICIDS_MAP, inplace=True)
    # Ensure we only select columns that exist after renaming
    existing_cols = [col for col in CICIDS_MAP.values() if col in df.columns]
    df = df[existing_cols]
    df['is_attack'] = df['attack_category'].apply(lambda x: 1 if x.strip().upper() != 'BENIGN' else 0)
    df['duration'] = df['duration'] / 1_000_000
    df['fwd_iat_mean'] = df['fwd_iat_mean'] / 1_000_000
    df['bwd_iat_mean'] = df['bwd_iat_mean'] / 1_000_000
    df['fwd_iat_std'] = df['fwd_iat_std'] / 1_000_000
    df['bwd_iat_std'] = df['bwd_iat_std'] / 1_000_000
    df['original_dataset'] = 'CICIDS2017'
    return df

def preprocess_unsw(df):
    """Preprocesses the UNSW_NB15 dataset."""
    df.rename(columns=UNSW_MAP, inplace=True)
    df = df[list(UNSW_MAP.values())]
    df['fwd_iat_mean'] = df['fwd_iat_mean'] / 1_000
    df['bwd_iat_mean'] = df['bwd_iat_mean'] / 1_000
    df['fwd_iat_std'] = df['fwd_iat_std'] / 1_000
    df['bwd_iat_std'] = df['bwd_iat_std'] / 1_000
    df['attack_category'] = df['attack_category'].str.strip()
    df.loc[df['attack_category'] == 'Backdoors', 'attack_category'] = 'Backdoor'
    df['original_dataset'] = 'UNSW_NB15'
    return df

def preprocess_ml_edge(df):
    """Preprocesses and aggregates the ML-EdgeIIoT dataset."""
    # Clean column names by stripping leading/trailing whitespace
    df.columns = df.columns.str.strip()
    df.rename(columns=ML_EDGE_MAP, inplace=True)
    # Ensure we only select columns that exist after renaming
    required_cols = list(ML_EDGE_MAP.values()) + ['frame.time_epoch', 'tcp.len', 'udp.length']
    existing_cols = [col for col in required_cols if col in df.columns]
    df = df[existing_cols]

    # Define a flow identifier
    flow_id = ['src_ip', 'dst_ip', 'src_port', 'dst_port']
    df_sorted = df.sort_values(by=flow_id + ['frame.time_epoch'])

    # Calculate packet-level features
    df_sorted['packet_len'] = df_sorted['tcp.len'].fillna(df_sorted['udp.length'].fillna(0))
    df_sorted['iat'] = df_sorted.groupby(flow_id)['frame.time_epoch'].diff()

    # Aggregate to flow-level
    agg_funcs = {
        'frame.time_epoch': ['min', 'max'],
        'packet_len': ['count', 'sum', 'mean'],
        'iat': ['mean', 'std'],
        'attack_category': 'first',
        'is_attack': 'first'
    }
    flow_df = df_sorted.groupby(flow_id).agg(agg_funcs).reset_index()

    # Flatten multi-index columns
    flow_df.columns = ['_'.join(col).strip() for col in flow_df.columns.values]

    # Rename and create unified columns
    flow_df.rename(columns={
        'src_ip_': 'src_ip',
        'dst_ip_': 'dst_ip',
        'src_port_': 'src_port',
        'dst_port_': 'dst_port',
        'frame.time_epoch_min': 'start_time',
        'frame.time_epoch_max': 'end_time',
        'packet_len_count': 'fwd_packets', # Simplified: assuming all packets are forward
        'packet_len_sum': 'fwd_bytes',
        'packet_len_mean': 'fwd_pkt_len_mean',
        'iat_mean': 'fwd_iat_mean',
        'iat_std': 'fwd_iat_std',
        'attack_category_first': 'attack_category',
        'is_attack_first': 'is_attack'
    }, inplace=True)

    flow_df['duration'] = flow_df['end_time'] - flow_df['start_time']
    flow_df['original_dataset'] = 'ML-EdgeIIoT'
    
    # Fill NaN for columns not present in ML-Edge
    flow_df['bwd_packets'] = 0
    flow_df['bwd_bytes'] = 0

    return flow_df

def main():
    """Loads, preprocesses, and merges the CICIDS and UNSW datasets based on common columns."""
    # Load CICIDS2017
    print(f"Loading {CICIDS_PATH}...")
    df_cicids = pd.read_csv(CICIDS_PATH, skipinitialspace=True, low_memory=False)
    print("Preprocessing CICIDS2017...")
    df_cicids_processed = preprocess_cicids(df_cicids.copy())

    # Load UNSW_NB15
    print(f"Loading {UNSW_PATH}...")
    df_unsw = pd.read_csv(UNSW_PATH, low_memory=False)
    print("Preprocessing UNSW_NB15...")
    df_unsw_processed = preprocess_unsw(df_unsw.copy())

    # Find common columns between the two processed datasets
    common_cols = list(set(df_cicids_processed.columns) & set(df_unsw_processed.columns))
    print(f"\nFound {len(common_cols)} common columns.")

    # Filter dataframes to only include common columns
    df_cicids_filtered = df_cicids_processed[common_cols]
    df_unsw_filtered = df_unsw_processed[common_cols]

    print("\nMerging CICIDS2017 and UNSW_NB15 datasets...")
    df_merged = pd.concat([df_cicids_filtered, df_unsw_filtered], ignore_index=True, sort=False)

    # Define a new output path for the focused dataset
    focused_output_path = 'unified_cicids_unsw_common.csv'
    print(f"Saving merged dataset to {focused_output_path}...")
    df_merged.to_csv(focused_output_path, index=False)

    print("\nMerge complete!")
    print(f"Total rows in merged dataset: {len(df_merged)}")
    print(f"Total columns in merged dataset: {len(df_merged.columns)}")
    print("\nCommon Columns:")
    print(df_merged.columns.tolist())
    print("\nDataset distribution:")
    print(df_merged['original_dataset'].value_counts())
    print("\nFirst 5 rows of merged data:")
    print(df_merged.head())

if __name__ == '__main__':
    main()
