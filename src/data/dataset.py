"""
Dataset classes for time series forecasting
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional

class TimeSeriesDataset(Dataset):
    """Dataset class for time series forecasting"""
    
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int, target_col_idx: int = 0):
        """
        Args:
            data: Input data array (samples, features)
            seq_len: Input sequence length
            pred_len: Prediction length
            target_col_idx: Index of target column in features
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col_idx = target_col_idx
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        # Input sequence
        x = self.data[idx:idx + self.seq_len]
        
        # Target sequence (only target column)
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_col_idx:self.target_col_idx+1]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

class MultiVariateTimeSeriesDataset(Dataset):
    """Dataset class for multivariate time series forecasting"""
    
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int, target_cols: list = None):
        """
        Args:
            data: Input data array (samples, features)
            seq_len: Input sequence length
            pred_len: Prediction length
            target_cols: List of target column indices
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_cols = target_cols if target_cols is not None else [0]
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        # Input sequence
        x = self.data[idx:idx + self.seq_len]
        
        # Target sequence (multiple target columns)
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_cols]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

def create_data_loaders(train_data: np.ndarray, valid_data: np.ndarray, test_data: np.ndarray,
                       seq_len: int, pred_len: int, batch_size: int = 32, 
                       target_col_idx: int = 0, shuffle_train: bool = True) -> Tuple:
    """Create data loaders for train, validation, and test sets"""
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len, target_col_idx)
    valid_dataset = TimeSeriesDataset(valid_data, seq_len, pred_len, target_col_idx)
    test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len, target_col_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader
