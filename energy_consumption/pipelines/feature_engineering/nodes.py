"""
This module contains the feature engineering nodes for the energy consumption pipeline.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_arima_data(split_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
    """
    Prepare data for ARIMA models.
    
    Args:
        split_data: Dictionary containing train and test data for each frequency
        
    Returns:
        Dictionary containing prepared data for ARIMA models
    """
    logger.info("Preparing data for ARIMA models")
    
    result = {}
    
    for freq, data in split_data.items():
        train = data["train"]
        test = data["test"]
        
        # ARIMA models work directly with the time series
        result[freq] = {
            "train": train,
            "test": test
        }
    
    return result

def prepare_ml_data(split_data: Dict[str, Dict[str, pd.DataFrame]], 
                   feature_data: pd.DataFrame,
                   lag_features: int = 7) -> Dict[str, Any]:
    """
    Prepare data for machine learning models (XGBoost, etc.).
    
    Args:
        split_data: Dictionary containing train and test data for each frequency
        feature_data: DataFrame with all features
        lag_features: Number of lag features to create
        
    Returns:
        Dictionary containing prepared data for ML models
    """
    logger.info(f"Preparing data for ML models with {lag_features} lag features")
    
    result = {}
    
    # Check if temperature data is available
    has_temperature = 'temperature' in feature_data.columns
    has_hdd_cdd = 'HDD' in feature_data.columns and 'CDD' in feature_data.columns
    
    for freq, data in split_data.items():
        # Get the target series
        train_series = data["train"]
        test_series = data["test"]
        
        # Create lag features
        train_df = pd.DataFrame(train_series)
        test_df = pd.DataFrame(test_series)
        
        # Add column name if it doesn't exist
        if train_df.columns[0] != 'target':
            train_df.columns = ['target']
            test_df.columns = ['target']
        
        # Create lag features
        for i in range(1, lag_features + 1):
            train_df[f'lag_{i}'] = train_df['target'].shift(i)
            test_df[f'lag_{i}'] = test_df['target'].shift(i)
        
        # Add temperature features if available
        if has_temperature or has_hdd_cdd:
            # Resample temperature data to match the frequency
            if freq == 'hourly_data':
                temp_resampled = feature_data.resample('H').mean()
            elif freq == 'daily_data':
                temp_resampled = feature_data.resample('D').mean()
            else:  # weekly_data
                temp_resampled = feature_data.resample('W').mean()
            
            # Add temperature
            if has_temperature:
                train_temp = temp_resampled.loc[train_df.index, 'temperature']
                test_temp = temp_resampled.loc[test_df.index, 'temperature']
                
                train_df['temperature'] = train_temp
                test_df['temperature'] = test_temp
                
                # Add temperature lag features
                for i in range(1, lag_features + 1):
                    train_df[f'temp_lag_{i}'] = train_df['temperature'].shift(i)
                    test_df[f'temp_lag_{i}'] = test_df['temperature'].shift(i)
            
            # Add HDD and CDD if available
            if has_hdd_cdd:
                train_hdd = temp_resampled.loc[train_df.index, 'HDD']
                train_cdd = temp_resampled.loc[train_df.index, 'CDD']
                test_hdd = temp_resampled.loc[test_df.index, 'HDD']
                test_cdd = temp_resampled.loc[test_df.index, 'CDD']
                
                train_df['HDD'] = train_hdd
                train_df['CDD'] = train_cdd
                test_df['HDD'] = test_hdd
                test_df['CDD'] = test_cdd
                
                # Add HDD and CDD lag features
                for i in range(1, lag_features + 1):
                    train_df[f'HDD_lag_{i}'] = train_df['HDD'].shift(i)
                    train_df[f'CDD_lag_{i}'] = train_df['CDD'].shift(i)
                    test_df[f'HDD_lag_{i}'] = test_df['HDD'].shift(i)
                    test_df[f'CDD_lag_{i}'] = test_df['CDD'].shift(i)
        
        # Add time-based features
        if freq == 'hourly_data':
            # Add hour of day, day of week features
            train_df['hour'] = train_df.index.hour
            train_df['day_of_week'] = train_df.index.dayofweek
            train_df['is_weekend'] = train_df.index.dayofweek.isin([5, 6]).astype(int)
            
            test_df['hour'] = test_df.index.hour
            test_df['day_of_week'] = test_df.index.dayofweek
            test_df['is_weekend'] = test_df.index.dayofweek.isin([5, 6]).astype(int)
            
            # Add cyclical features
            train_df['hour_sin'] = np.sin(2 * np.pi * train_df['hour']/24)
            train_df['hour_cos'] = np.cos(2 * np.pi * train_df['hour']/24)
            train_df['day_of_week_sin'] = np.sin(2 * np.pi * train_df['day_of_week']/7)
            train_df['day_of_week_cos'] = np.cos(2 * np.pi * train_df['day_of_week']/7)
            
            test_df['hour_sin'] = np.sin(2 * np.pi * test_df['hour']/24)
            test_df['hour_cos'] = np.cos(2 * np.pi * test_df['hour']/24)
            test_df['day_of_week_sin'] = np.sin(2 * np.pi * test_df['day_of_week']/7)
            test_df['day_of_week_cos'] = np.cos(2 * np.pi * test_df['day_of_week']/7)
        
        elif freq == 'daily_data':
            # Add day of week, day of month, month features
            train_df['day_of_week'] = train_df.index.dayofweek
            train_df['day_of_month'] = train_df.index.day
            train_df['month'] = train_df.index.month
            train_df['is_weekend'] = train_df.index.dayofweek.isin([5, 6]).astype(int)
            
            test_df['day_of_week'] = test_df.index.dayofweek
            test_df['day_of_month'] = test_df.index.day
            test_df['month'] = test_df.index.month
            test_df['is_weekend'] = test_df.index.dayofweek.isin([5, 6]).astype(int)
            
            # Add cyclical features
            train_df['day_of_week_sin'] = np.sin(2 * np.pi * train_df['day_of_week']/7)
            train_df['day_of_week_cos'] = np.cos(2 * np.pi * train_df['day_of_week']/7)
            train_df['day_of_month_sin'] = np.sin(2 * np.pi * train_df['day_of_month']/31)
            train_df['day_of_month_cos'] = np.cos(2 * np.pi * train_df['day_of_month']/31)
            train_df['month_sin'] = np.sin(2 * np.pi * train_df['month']/12)
            train_df['month_cos'] = np.cos(2 * np.pi * train_df['month']/12)
            
            test_df['day_of_week_sin'] = np.sin(2 * np.pi * test_df['day_of_week']/7)
            test_df['day_of_week_cos'] = np.cos(2 * np.pi * test_df['day_of_week']/7)
            test_df['day_of_month_sin'] = np.sin(2 * np.pi * test_df['day_of_month']/31)
            test_df['day_of_month_cos'] = np.cos(2 * np.pi * test_df['day_of_month']/31)
            test_df['month_sin'] = np.sin(2 * np.pi * test_df['month']/12)
            test_df['month_cos'] = np.cos(2 * np.pi * test_df['month']/12)
        
        elif freq == 'weekly_data':
            # Add week of year, month features
            train_df['week_of_year'] = train_df.index.isocalendar().week
            train_df['month'] = train_df.index.month
            
            test_df['week_of_year'] = test_df.index.isocalendar().week
            test_df['month'] = test_df.index.month
            
            # Add cyclical features
            train_df['week_of_year_sin'] = np.sin(2 * np.pi * train_df['week_of_year']/52)
            train_df['week_of_year_cos'] = np.cos(2 * np.pi * train_df['week_of_year']/52)
            train_df['month_sin'] = np.sin(2 * np.pi * train_df['month']/12)
            train_df['month_cos'] = np.cos(2 * np.pi * train_df['month']/12)
            
            test_df['week_of_year_sin'] = np.sin(2 * np.pi * test_df['week_of_year']/52)
            test_df['week_of_year_cos'] = np.cos(2 * np.pi * test_df['week_of_year']/52)
            test_df['month_sin'] = np.sin(2 * np.pi * test_df['month']/12)
            test_df['month_cos'] = np.cos(2 * np.pi * test_df['month']/12)
        
        # Drop rows with NaN values
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        
        # Split features and target
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame to keep column names
        X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
        
        result[freq] = {
            "X_train": X_train_scaled,
            "y_train": y_train,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "scaler": scaler,
            "feature_names": list(X_train.columns)
        }
    
    return result

def create_sequences(split_data: Dict[str, Dict[str, pd.DataFrame]], 
                    feature_data: pd.DataFrame,
                    seq_length: int = 24) -> Dict[str, Any]:
    """
    Create sequences for RNN training.
    
    Args:
        split_data: Dictionary containing train and test data for each frequency
        feature_data: DataFrame with all features
        seq_length: Length of the sequences
        
    Returns:
        Dictionary containing sequences for RNN models
    """
    logger.info(f"Creating sequences for RNN models with sequence length {seq_length}")
    
    result = {}
    
    for freq, data in split_data.items():
        train_series = data["train"]
        test_series = data["test"]
        
        # Create sequences
        train_sequences = []
        train_targets = []
        
        for i in range(len(train_series) - seq_length):
            train_sequences.append(train_series.iloc[i:i+seq_length].values)
            train_targets.append(train_series.iloc[i+seq_length])
        
        test_sequences = []
        test_targets = []
        
        for i in range(len(test_series) - seq_length):
            test_sequences.append(test_series.iloc[i:i+seq_length].values)
            test_targets.append(test_series.iloc[i+seq_length])
        
        # Convert to numpy arrays
        X_train = np.array(train_sequences).reshape(-1, seq_length, 1)
        y_train = np.array(train_targets)
        X_test = np.array(test_sequences).reshape(-1, seq_length, 1)
        y_test = np.array(test_targets)
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Flatten X_train for scaling
        X_train_flat = X_train.reshape(-1, 1)
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        
        # Use the same scaler for X_test
        X_test_flat = X_test.reshape(-1, 1)
        X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
        
        # Scale y_train and y_test
        y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        result[freq] = {
            "X_train": X_train_scaled,
            "y_train": y_train_scaled,
            "X_test": X_test_scaled,
            "y_test": y_test_scaled,
            "scaler": scaler,
            "original_y_test": y_test
        }
    
    return result

def create_sequences_with_temperature(split_data: Dict[str, Dict[str, pd.DataFrame]], 
                                     feature_data: pd.DataFrame,
                                     seq_length: int = 24) -> Dict[str, Any]:
    """
    Create sequences for RNN training with temperature data.
    
    Args:
        split_data: Dictionary containing train and test data for each frequency
        feature_data: DataFrame with all features
        seq_length: Length of the sequences
        
    Returns:
        Dictionary containing sequences for RNN models
    """
    logger.info(f"Creating sequences for RNN models with sequence length {seq_length} and temperature data")
    
    result = {}
    
    # Check if temperature data is available
    has_temperature = 'temperature' in feature_data.columns
    
    for freq, data in split_data.items():
        train_series = data["train"]
        test_series = data["test"]
        
        # Resample temperature data to match the frequency
        if has_temperature:
            if freq == 'hourly_data':
                temp_resampled = feature_data['temperature'].resample('H').mean()
            elif freq == 'daily_data':
                temp_resampled = feature_data['temperature'].resample('D').mean()
            else:  # weekly_data
                temp_resampled = feature_data['temperature'].resample('W').mean()
            
            # Align indices
            temp_train = temp_resampled[temp_resampled.index.isin(train_series.index)]
            temp_test = temp_resampled[temp_resampled.index.isin(test_series.index)]
            
            # Create sequences with both consumption and temperature
            train_sequences = []
            train_targets = []
            
            for i in range(len(train_series) - seq_length):
                # Get consumption sequence
                consumption_seq = train_series.iloc[i:i+seq_length].values
                # Get temperature sequence
                temp_seq = temp_train.iloc[i:i+seq_length].values if i+seq_length <= len(temp_train) else None
                
                if temp_seq is not None and not np.isnan(temp_seq).any():
                    # Combine consumption and temperature
                    combined_seq = np.column_stack((consumption_seq, temp_seq))
                    train_sequences.append(combined_seq)
                    train_targets.append(train_series.iloc[i+seq_length])
            
            test_sequences = []
            test_targets = []
            
            for i in range(len(test_series) - seq_length):
                # Get consumption sequence
                consumption_seq = test_series.iloc[i:i+seq_length].values
                # Get temperature sequence
                temp_seq = temp_test.iloc[i:i+seq_length].values if i+seq_length <= len(temp_test) else None
                
                if temp_seq is not None and not np.isnan(temp_seq).any():
                    # Combine consumption and temperature
                    combined_seq = np.column_stack((consumption_seq, temp_seq))
                    test_sequences.append(combined_seq)
                    test_targets.append(test_series.iloc[i+seq_length])
            
            # Convert to numpy arrays
            X_train = np.array(train_sequences)
            y_train = np.array(train_targets)
            X_test = np.array(test_sequences)
            y_test = np.array(test_targets)
            
            # Scale the data
            # For multivariate sequences, we need to reshape carefully
            scaler = MinMaxScaler(feature_range=(0, 1))
            
            # Reshape for scaling
            n_samples_train, n_timesteps, n_features = X_train.shape
            X_train_reshaped = X_train.reshape(-1, n_features)
            X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(n_samples_train, n_timesteps, n_features)
            
            n_samples_test, _, _ = X_test.shape
            X_test_reshaped = X_test.reshape(-1, n_features)
            X_test_scaled = scaler.transform(X_test_reshaped).reshape(n_samples_test, n_timesteps, n_features)
            
            # Scale y_train and y_test
            y_scaler = MinMaxScaler(feature_range=(0, 1))
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
            
            result[freq] = {
                "X_train": X_train_scaled,
                "y_train": y_train_scaled,
                "X_test": X_test_scaled,
                "y_test": y_test_scaled,
                "scaler": scaler,
                "y_scaler": y_scaler,
                "original_y_test": y_test,
                "has_temperature": True,
                "n_features": n_features
            }
        else:
            # Fall back to original sequence creation if no temperature data
            result[freq] = create_sequences(split_data, feature_data, seq_length)[freq]
    
    return result

def prepare_prophet_data(split_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
    """
    Prepare data for Prophet models.
    
    Args:
        split_data: Dictionary containing train and test data for each frequency
        
    Returns:
        Dictionary containing prepared data for Prophet models
    """
    logger.info("Preparing data for Prophet models")
    
    result = {}
    
    for freq, data in split_data.items():
        train = data["train"]
        test = data["test"]
        
        # Prophet requires a specific DataFrame format with 'ds' and 'y' columns
        train_df = pd.DataFrame({'ds': train.index, 'y': train.values})
        test_df = pd.DataFrame({'ds': test.index, 'y': test.values})
        
        result[freq] = {
            "train": train_df,
            "test": test_df
        }
    
    return result