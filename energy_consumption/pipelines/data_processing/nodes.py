"""
This module contains the data processing nodes for the energy consumption pipeline.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import requests
from io import StringIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_data(url: str) -> pd.DataFrame:
    """
    Download data from the specified URL.
    
    Args:
        url: URL to download the data from
        
    Returns:
        DataFrame containing the raw data
    """
    logger.info(f"Downloading data from {url}")
    response = requests.get(url)
    data = StringIO(response.text)
    df = pd.read_csv(data)
    logger.info(f"Downloaded data with shape: {df.shape}")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw data.
    
    Args:
        df: Raw data DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Preprocessing data")
    
    # Convert datetime column to datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Drop the unnamed column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Convert columns to numeric
    numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                      'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    
    logger.info(f"Preprocessed data with shape: {df.shape}")
    return df

def download_temperature_data(url: str) -> pd.DataFrame:
    """
    Download temperature data from the specified URL.
    
    Args:
        url: URL to download the temperature data from
        
    Returns:
        DataFrame containing the temperature data
    """
    logger.info(f"Downloading temperature data from {url}")
    response = requests.get(url)
    data = StringIO(response.text)
    df = pd.read_csv(data)
    logger.info(f"Downloaded temperature data with shape: {df.shape}")
    return df

def preprocess_temperature_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the temperature data.
    
    Args:
        df: Raw temperature data DataFrame
        
    Returns:
        Preprocessed temperature DataFrame
    """
    logger.info("Preprocessing temperature data")
    
    # Convert datetime column to datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Drop unnecessary columns if they exist
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    
    logger.info(f"Preprocessed temperature data with shape: {df.shape}")
    return df

def merge_energy_temperature_data(energy_df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge energy consumption data with temperature data.
    
    Args:
        energy_df: Energy consumption DataFrame
        temp_df: Temperature DataFrame
        
    Returns:
        Merged DataFrame
    """
    logger.info("Merging energy and temperature data")
    
    # Ensure both DataFrames have datetime index
    if not isinstance(energy_df.index, pd.DatetimeIndex):
        energy_df['datetime'] = pd.to_datetime(energy_df['datetime'])
        energy_df.set_index('datetime', inplace=True)
    
    if not isinstance(temp_df.index, pd.DatetimeIndex):
        temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
        temp_df.set_index('datetime', inplace=True)
    
    # Merge the DataFrames on the datetime index
    merged_df = pd.merge(energy_df, temp_df, left_index=True, right_index=True, how='left')
    
    # Rename temperature columns for clarity
    for col in temp_df.columns:
        if col in energy_df.columns:
            merged_df.rename(columns={f"{col}_y": f"temp_{col}"}, inplace=True)
            merged_df.rename(columns={f"{col}_x": col}, inplace=True)
    
    # Fill missing temperature values using forward fill
    merged_df.fillna(method='ffill', inplace=True)
    
    # Calculate heating and cooling degree days
    if 'temperature' in merged_df.columns:
        # Base temperature for HDD and CDD (commonly used values)
        base_temp_heating = 18.0  # 18°C or ~65°F
        base_temp_cooling = 22.0  # 22°C or ~72°F
        
        # Calculate HDD and CDD
        merged_df['HDD'] = merged_df['temperature'].apply(lambda x: max(0, base_temp_heating - x))
        merged_df['CDD'] = merged_df['temperature'].apply(lambda x: max(0, x - base_temp_cooling))
    
    logger.info(f"Merged data with shape: {merged_df.shape}")
    return merged_df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to the DataFrame.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        DataFrame with additional time features
    """
    logger.info("Adding time features")
    
    # Add time-based features
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    df['season'] = df['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                             'Spring' if x in [3, 4, 5] else
                                             'Summer' if x in [6, 7, 8] else 'Fall')
    
    # Add cyclical features for hour, day, month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # Calculate total consumption
    df['total_metering'] = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
    df['other_consumption'] = df['Global_active_power']*1000/60 - df['total_metering']
    
    logger.info(f"Added time features to data. New shape: {df.shape}")
    return df

def create_resampled_datasets(df: pd.DataFrame, target_col: str) -> Dict[str, pd.DataFrame]:
    """
    Create resampled datasets at different frequencies.
    
    Args:
        df: DataFrame with features
        target_col: Target column name
        
    Returns:
        Dictionary containing resampled DataFrames
    """
    logger.info("Creating resampled datasets")
    
    # Create hourly, daily, and weekly resampled data
    hourly_data = df[target_col].resample('H').mean()
    daily_data = df[target_col].resample('D').mean()
    weekly_data = df[target_col].resample('W').mean()
    
    logger.info(f"Created resampled datasets: hourly ({len(hourly_data)}), daily ({len(daily_data)}), weekly ({len(weekly_data)})")
    
    return {
        "hourly_data": hourly_data,
        "daily_data": daily_data,
        "weekly_data": weekly_data
    }

def split_data(data: Dict[str, pd.DataFrame], test_size: float = 0.2) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split the resampled datasets into training and testing sets.
    
    Args:
        data: Dictionary containing resampled DataFrames
        test_size: Proportion of data to use for testing
        
    Returns:
        Dictionary containing train and test DataFrames for each frequency
    """
    logger.info(f"Splitting data with test_size={test_size}")
    
    result = {}
    
    for key, series in data.items():
        # Sort by datetime to ensure chronological order
        series = series.sort_index()
        
        # Calculate split point
        split_idx = int(len(series) * (1 - test_size))
        
        # Split the data
        train = series.iloc[:split_idx]
        test = series.iloc[split_idx:]
        
        logger.info(f"{key}: Train data shape: {train.shape}, Test data shape: {test.shape}")
        
        result[key] = {
            "train": train,
            "test": test
        }
    
    return result