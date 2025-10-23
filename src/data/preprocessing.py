"""
Data preprocessing module for time series forecasting
Handles data loading, resampling to 15-minute timeframe, and technical indicators
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class CryptoDataPreprocessor:
    """Preprocessor for cryptocurrency time series data"""
    
    def __init__(self, data_dir: str, target_col: str = 'Close'):
        self.data_dir = data_dir
        self.target_col = target_col
        
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Load raw crypto data from CSV file"""
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        return df
    
    def resample_to_15min(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample data to 15-minute timeframe using OHLCV aggregation"""
        # Resample to 15-minute intervals
        df_15min = df.resample('15T').agg({
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        print(f"Resampled to 15-minute data: {len(df_15min)} samples")
        return df_15min
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators to the dataset"""
        df = df.copy()
        
        # Price-based indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        df['EMA_5'] = df['Close'].ewm(span=5).mean()
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        # Commodity Channel Index (CCI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price change indicators
        df['Price_change'] = df['Close'].pct_change()
        df['Price_change_5'] = df['Close'].pct_change(periods=5)
        df['Price_change_10'] = df['Close'].pct_change(periods=10)
        
        # Volatility
        df['Volatility'] = df['Price_change'].rolling(window=20).std()
        
        # Momentum indicators
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Rate of Change (ROC)
        df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
        df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
        
        print(f"Added {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} technical indicators")
        return df
    
    def split_data_by_years(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train (<=2023), validation (2024), test (2025) sets"""
        train_data = df[df.index.year <= 2023]
        valid_data = df[df.index.year == 2024]
        test_data = df[df.index.year == 2025]
        
        print(f"Train data: {len(train_data)} samples ({train_data.index.min()} to {train_data.index.max()})")
        print(f"Valid data: {len(valid_data)} samples ({valid_data.index.min()} to {valid_data.index.max()})")
        print(f"Test data: {len(test_data)} samples ({test_data.index.min()} to {test_data.index.max()})")
        
        return train_data, valid_data, test_data
    
    def prepare_features(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """Prepare feature matrix from dataframe"""
        if feature_cols is None:
            # Use all numeric columns except the target
            feature_cols = [col for col in df.columns if col != self.target_col and df[col].dtype in ['float64', 'int64']]
        
        # Select features and handle missing values
        features = df[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return features.values
    
    def process_single_file(self, file_path: str, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a single crypto file through the complete pipeline"""
        # Load and resample data
        df = self.load_raw_data(file_path)
        df_15min = self.resample_to_15min(df)
        
        # Add technical indicators
        df_with_indicators = self.add_technical_indicators(df_15min)
        
        # Split by years
        train_data, valid_data, test_data = self.split_data_by_years(df_with_indicators)
        
        # Limit samples if specified
        if max_samples:
            if len(train_data) > max_samples:
                train_data = train_data.tail(max_samples)
            if len(valid_data) > max_samples // 4:
                valid_data = valid_data.tail(max_samples // 4)
            if len(test_data) > max_samples // 4:
                test_data = test_data.tail(max_samples // 4)
        
        # Prepare features (use all available features)
        train_features = self.prepare_features(train_data)
        valid_features = self.prepare_features(valid_data)
        test_features = self.prepare_features(test_data)
        
        return train_features, valid_features, test_features
    
    def get_all_crypto_files(self) -> List[str]:
        """Get list of all crypto CSV files in the data directory"""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        csv_files.sort()
        return csv_files
