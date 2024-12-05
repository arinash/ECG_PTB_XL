import os
import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy.signal import butter, filtfilt, medfilt


def load_raw_data(df, sampling_rate, base_path):
    """
    Load raw ECG signal data from PTB-XL dataset.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing filenames of ECG data.
        sampling_rate (int): Sampling rate to use (100 or 500 Hz).
        base_path (str): Base path to the PTB-XL dataset.
    
    Returns:
        np.ndarray: Array of ECG signals.
    """
    if sampling_rate == 100:
        filenames = df['filename_lr']
        records_folder = 'records100'
    else:
        filenames = df['filename_hr']
        records_folder = 'records500'
    
    # Load ECG signals
    print(f"Loading raw data from {records_folder}...")
    data = []
    for idx, f in enumerate(filenames):
        if idx % 100 == 0:
            print(f"Processing {idx+1}/{len(filenames)} files...")
        
        #file_path = os.path.join(base_path, f"{f[:-4]}")
        file_path = os.path.join(base_path, f)
        
        # Read signal data
        data.append(wfdb.rdsamp(file_path))
        
    signals = np.array([signal for signal, meta in data])
    print(f"Finished loading {len(filenames)} raw ECG signals.")
    return signals


def aggregate_diagnostic(y_dic, agg_df):
    """
    Aggregate diagnostic classes using scp_statements.csv mapping.
    
    Parameters:
        y_dic (dict): Dictionary of SCP codes and likelihoods for a record.
        agg_df (pd.DataFrame): Aggregation DataFrame with diagnostic mappings.
    
    Returns:
        list: List of diagnostic superclasses for the record.
    """
    diagnostics = []
    for key in y_dic.keys():
        if key in agg_df.index:
            diagnostics.append(agg_df.loc[key]['diagnostic_class'])
    return list(set(diagnostics))


def normalize_signals(signals):
    """
    Normalize ECG signals using Min-Max scaling.
    
    Parameters:
        signals (np.ndarray): Raw ECG signals to normalize.
    
    Returns:
        np.ndarray: Normalized ECG signals.
    """
    print("Normalizing ECG signals...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    num_samples, num_leads, num_timesteps = signals.shape
    signals_reshaped = signals.reshape(-1, num_timesteps)
    signals_normalized = scaler.fit_transform(signals_reshaped)
    print("Normalization complete.")
    return signals_normalized.reshape(num_samples, num_leads, num_timesteps)


def remove_baseline_drift(signal, sampling_rate=100, cutoff=0.5):
    """
    Removes baseline drift using a high-pass Butterworth filter.
    
    Parameters:
        signal (np.ndarray): The ECG signal.
        sampling_rate (int): Sampling rate of the ECG signal.
        cutoff (float): Cutoff frequency for the high-pass filter.
    
    Returns:
        np.ndarray: Signal with baseline drift removed.
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal, axis=0)


def remove_static_noise(signal, sampling_rate=100, cutoff=40):
    """
    Removes static noise using a low-pass Butterworth filter.
    
    Parameters:
        signal (np.ndarray): The ECG signal.
        sampling_rate (int): Sampling rate of the ECG signal.
        cutoff (float): Cutoff frequency for the low-pass filter.
    
    Returns:
        np.ndarray: Signal with high-frequency noise removed.
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal, axis=0)


def remove_burst_noise(signal, kernel_size=5):
    """
    Removes burst noise using median filtering.
    
    Parameters:
        signal (np.ndarray): The ECG signal.
        kernel_size (int): Kernel size for the median filter.
    
    Returns:
        np.ndarray: Signal with spikes removed.
    """
    return medfilt(signal, kernel_size=(kernel_size, 1))


def reduce_noise(signals, metadata, sampling_rate=100):
    """
    Preprocess ECG signals by reducing noise.

    Parameters:
        signals (np.ndarray): Raw ECG signals.
        metadata (pd.DataFrame): Metadata containing noise information.
        sampling_rate (int): Sampling rate of the ECG signal.

    Returns:
        np.ndarray: Preprocessed ECG signals.
    """
    preprocessed_signals = []
    for i, signal in enumerate(signals):
        # Baseline drift removal
        if pd.notnull(metadata.iloc[i]['baseline_drift']):
            signal = remove_baseline_drift(signal, sampling_rate)

        # Static noise removal
        if pd.notnull(metadata.iloc[i]['static_noise']):
            signal = remove_static_noise(signal, sampling_rate)

        # Burst noise removal
        if pd.notnull(metadata.iloc[i]['burst_noise']):
            signal = remove_burst_noise(signal)

        preprocessed_signals.append(signal)
    
    return np.array(preprocessed_signals)


def impute_missing_values(df, columns):
    """
    Impute missing values in specified columns using mean imputation.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing metadata.
        columns (list): List of columns to impute missing values.
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    print(f"Imputing missing values for columns: {columns}...")
    imputer = SimpleImputer(strategy='mean')
    df[columns] = imputer.fit_transform(df[columns])
    print(f"Imputation for columns {columns} complete.")
    return df

if __name__ == "__main__":

    BASE_PATH = '../data/'
    SAMPLING_RATE = 100

    # Load the metadata CSV
    print("Loading metadata CSV (ptbxl_database.csv)...")
    metadata_df = pd.read_csv(os.path.join(BASE_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
    metadata_df['scp_codes'] = metadata_df['scp_codes'].apply(lambda x: ast.literal_eval(x))

    # Load scp_statements.csv for diagnostic aggregation
    print("Loading scp_statements.csv...")
    scp_statements_df = pd.read_csv(os.path.join(BASE_PATH, 'scp_statements.csv'), index_col=0)
    scp_statements_df = scp_statements_df[scp_statements_df['diagnostic'] == 1]

    # Aggregate diagnostic superclass labels
    print("Aggregating diagnostic superclass labels...")
    metadata_df['diagnostic_superclass'] = metadata_df['scp_codes'].apply(
        lambda x: aggregate_diagnostic(x, scp_statements_df)
    )

    # Handle missing values in height and weight
    metadata_df = impute_missing_values(metadata_df, ['height', 'weight'])

    # Load raw waveform data
    X = load_raw_data(metadata_df, SAMPLING_RATE, BASE_PATH)

    # Normalize signals
    X_normalized = normalize_signals(X)
    X_preprocessed = reduce_noise(X, metadata_df)

    # Split data into train, validation, and test sets based on fold
    TRAIN_FOLDS = range(1, 9)  # Folds 1-8 for training
    VALIDATION_FOLD = 9        # Fold 9 for validation
    TEST_FOLD = 10             # Fold 10 for testing

    # Training data
    train_mask = metadata_df['strat_fold'].isin(TRAIN_FOLDS)
    X_train = X_preprocessed[train_mask]
    y_train = metadata_df[train_mask]['diagnostic_superclass']

    # Validation data
    validation_mask = metadata_df['strat_fold'] == VALIDATION_FOLD
    X_val = X_preprocessed[validation_mask]
    y_val = metadata_df[validation_mask]['diagnostic_superclass']

    # Testing data
    test_mask = metadata_df['strat_fold'] == TEST_FOLD
    X_test = X_preprocessed[test_mask]
    y_test = metadata_df[test_mask]['diagnostic_superclass']

    # Save preprocessed data
    os.makedirs('../data/processed', exist_ok=True)
    print("Saving preprocessed data...")
    np.save('../data/processed/X_train.npy', X_train)
    np.save('../data/processed/X_val.npy', X_val)
    np.save('../data/processed/X_test.npy', X_test)
    y_train.to_pickle('../data/processed/y_train.pkl')
    y_val.to_pickle('../data/processed/y_val.pkl')
    y_test.to_pickle('../data/processed/y_test.pkl')

    print("Data preprocessing completed. Normalized signals and splits saved.")