
import pandas as pd
import holidays
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# Initialize French holiday calendar
fr_holidays = holidays.France(years=range(2006, 2011))

# Get all holiday dates in the period
holiday_dates = list(fr_holidays.keys())

class DataLoader:
    def __init__(self):
        self.file_id = "1c222AbSUMn9vKcepLZDnyCKUN2B8BQtP"
        self.url = f"https://drive.google.com/uc?export=download&id={self.file_id}"
        DataLoader.load_data_Modeling(self.url)
        DataLoader.load_data_Eda(self.url)

    @staticmethod
    def basic_features_engineering(df):
        # Ensure the index is a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Basic temporal features
        df['Day_of_week'] = df.index.dayofweek
        df['Day_of_month'] = df.index.day
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        df['Is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        # Make seasonal feature
        df['Is_summer'] = ((df.index.month >= 6) & (df.index.month <= 8)).astype(int)
        df['Is_winter'] = ((df.index.month == 12) | (df.index.month <= 2)).astype(int)
        df['Is_spring'] = ((df.index.month >= 3) & (df.index.month <= 5)).astype(int)
        df['Is_autumn'] = ((df.index.month >= 9) & (df.index.month <= 11)).astype(int)

        # Ensure holiday_dates is defined (replace with actual holiday list)
        holiday_dates = pd.to_datetime(["2025-01-01", "2025-12-25"])  # Example holidays
        df['Is_holiday'] = df.index.isin(holiday_dates).astype(int)

        return df

    @staticmethod
    def load_data_Eda(url):
        df = pd.read_csv(url, sep=',')
        df['Datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('Datetime', inplace=True)

        # Handle missing values
        df['Global_active_power'].interpolate(method='linear', inplace=True)
        df.dropna(subset=['Global_active_power'], inplace=True)

        df = DataLoader.basic_features_engineering(df)

        return df

    @staticmethod
    def load_data_Modeling(url):
        df = pd.read_csv(url, sep=',')
        df['Datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('Datetime', inplace=True)

        # Handle missing values
        df['Global_active_power'].interpolate(method='linear', inplace=True)
        df.dropna(subset=['Global_active_power'], inplace=True)

        # Basic temporal features
        df = DataLoader.basic_features_engineering(df)

        # Cyclical encoding for seasons
        df['Sin_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['Cos_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365)

        # Lag features (autoregressive)
        for lag in [1, 2, 7, 14, 30]:  # daily, weekly, monthly patterns
            df[f'Global_lag_{lag}d'] = df['Global_active_power'].shift(lag)
            if 'temp' in df.columns:
                df[f'Temp_lag_{lag}d'] = df['temp'].shift(lag)
            if 'humidity' in df.columns:
                df[f'Humidity_lag_{lag}d'] = df['humidity'].shift(lag)

        # Rolling statistics
        df['GAP_rolling_7d_mean'] = df['Global_active_power'].shift(1).rolling(7).mean()
        df['GAP_rolling_30d_mean'] = df['Global_active_power'].shift(1).rolling(30).mean()
        df['GAP_rolling_7d_std'] = df['Global_active_power'].shift(1).rolling(7).std()
        df['GAP_rolling_30d_std'] = df['Global_active_power'].shift(1).rolling(30).std()

        # Weather interactions
        if 'temp' in df.columns and 'humidity' in df.columns:
            df['Temp_humidity'] = df['temp'] * df['humidity']

        # Rolling statistics for weather
        if 'temp' in df.columns:
            df['Temp_rolling_7d_mean'] = df['temp'].shift(1).rolling(7).mean()
            df['Temp_rolling_30d_mean'] = df['temp'].shift(1).rolling(30).mean()
            df['Temp_rolling_7d_std'] = df['temp'].shift(1).rolling(7).std()
            df['Temp_rolling_30d_std'] = df['temp'].shift(1).rolling(30).std()

        # Growth rate
        df['GAP_Growth_7d'] = df['Global_active_power'].pct_change(7)

        return df.dropna()
    
    
class OutlaierDetector:
    def __init__(self):
        self.series = None
        self.method= 'iqr'
        self.window = 30
        self.threshold = 3
        self.outliers = None
        self.lower = 0.05
        self.upper = 0.95

    @staticmethod
    def detect_outliers(series, method='iqr', window=30, threshold=3):
        if method == 'iqr':
            # IQR method (robust for skewed distributions)
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            # Z-score method (for normal distributions)
            z = np.abs(stats.zscore(series))
            return z > threshold
        
        elif method == 'rolling':
            # Time-aware rolling z-score
            mean = series.rolling(window=window, min_periods=1).mean()
            std = series.rolling(window=window, min_periods=1).std()
            z = np.abs((series - mean) / std)
            return z > threshold
        
        elif method == 'mad':
            # Median Absolute Deviation (robust)
            median = series.median()
            mad = np.abs(series - median).median()
            modified_z = 0.6745 * (series - median) / mad
            return np.abs(modified_z) > threshold
        
        else:
            raise ValueError("Invalid method. Choose: 'iqr', 'zscore', 'rolling', 'mad'")

    @staticmethod
    def plot_outliers(series, outliers):
        plt.figure(figsize=(14, 6))
        plt.plot(series, label='Original')
        plt.scatter(outliers[outliers].index, 
                    series[outliers], 
                    color='red', label='Outliers')
        plt.title('Outlier Detection')
        plt.legend()
        plt.show()

    def handle_outliers(self, series, outliers, method='winsorize', **kwargs):
        """
        Handle detected outliers
        """
        if not isinstance(series, pd.Series):
            raise TypeError("Expected a Pandas Series, but got a different type.")

        cleaned = series.copy()
        
        if method == 'winsorize':
            # Cap extreme values at percentiles
            lower = kwargs.get('lower', 0.05)
            upper = kwargs.get('upper', 0.95)
            lower_bound = series.quantile(lower)
            upper_bound = series.quantile(upper)
            cleaned[outliers & (series < lower_bound)] = lower_bound
            cleaned[outliers & (series > upper_bound)] = upper_bound
        elif method == 'impute':
            # Time-aware imputation
            strategy = kwargs.get('strategy', 'linear')
            cleaned[outliers] = np.nan
            return cleaned.interpolate(method=strategy)
        elif method == 'median':
            # Rolling median replacement
            window = kwargs.get('window', 5)
            cleaned[outliers] = np.nan
            return cleaned.fillna(cleaned.rolling(window, min_periods=1).median())
        elif method == 'remove':
            # Remove outliers (use with caution)
            return series[~outliers]
        else:
            raise ValueError("Invalid method. Choose: 'winsorize', 'impute', 'median', 'remove'")
        return cleaned

if __name__ == "__main__":
    outlier = OutlaierDetector()
    # Example usage of DataLoader
    outliers= outlier.detect_outliers(pd.Series([1, 2, 3, 100, 5, 6]), method='iqr', threshold=1.5, window=30)
    print("Detected outliers:", outliers)
