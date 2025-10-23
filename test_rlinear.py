import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our RLinear models
from rlinear_model import RLinearModel, RLinearOLS
from utils.metrics import MAE, MSE, RMSE

class CryptoDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

def load_crypto_data(file_path, target_col='Close', max_samples=50000):
    """Load crypto data and return train/valid/test splits"""
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Select target column (Close price)
    data = df[target_col].values.reshape(-1, 1)
    
    # Split data based on years
    train_data = data[df.index.year <= 2023]
    valid_data = data[df.index.year == 2024]
    test_data = data[df.index.year == 2025]
    
    # Limit data for testing if too large
    if len(train_data) > max_samples:
        train_data = train_data[-max_samples:]
    if len(valid_data) > max_samples // 4:
        valid_data = valid_data[-max_samples // 4:]
    if len(test_data) > max_samples // 4:
        test_data = test_data[-max_samples // 4:]
    
    print(f"Train data: {len(train_data)} samples")
    print(f"Valid data: {len(valid_data)} samples") 
    print(f"Test data: {len(test_data)} samples")
    
    return train_data, valid_data, test_data

def calculate_trend_match(pred, true):
    """Calculate trend match percentage"""
    pred_diff = np.diff(pred, axis=1)
    true_diff = np.diff(true, axis=1)
    
    # Calculate trend direction matches
    pred_trend = np.sign(pred_diff)
    true_trend = np.sign(true_diff)
    
    matches = (pred_trend == true_trend).astype(int)
    trend_match = np.mean(matches) * 100
    
    # Handle NaN case
    if np.isnan(trend_match):
        trend_match = 0.0
    
    return trend_match

def run_rlinear_ols_experiment(data_path, configs):
    """Run RLinear OLS experiment for a single crypto pair"""
    print(f"\n{'='*60}")
    print(f"Running RLinear OLS on {os.path.basename(data_path)}")
    print(f"{'='*60}")
    
    # Load data
    train_data, valid_data, test_data = load_crypto_data(data_path)
    
    # Check if we have enough data
    if len(train_data) < configs.seq_len + configs.pred_len:
        print(f"Not enough training data: {len(train_data)} samples")
        return None
    
    if len(test_data) < configs.seq_len + configs.pred_len:
        print(f"Not enough test data: {len(test_data)} samples")
        return None
    
    # Normalize data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    # Create RLinear OLS model
    print("Training RLinear OLS...")
    model = RLinearOLS(train_data_scaled, 
                      configs.seq_len, 
                      configs.pred_len,
                      instance_norm=True,
                      individual=configs.individual,
                      alpha=0.000001,
                      verbose=False)
    
    # Create test instances
    test_instances = np.lib.stride_tricks.sliding_window_view(test_data_scaled, (configs.seq_len + configs.pred_len), axis=0)
    X_test = test_instances[:, :, :configs.seq_len]
    y_test = test_instances[:, :, configs.seq_len:]
    
    # Make predictions
    print("Evaluating RLinear OLS...")
    preds = model.predict(X_test)
    
    # Inverse transform predictions and true values
    preds_original = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_original = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    # Calculate metrics
    mae = MAE(preds_original, trues_original)
    mse = MSE(preds_original, trues_original)
    rmse = RMSE(preds_original, trues_original)
    trend_match = calculate_trend_match(preds_original, trues_original)
    
    return {
        'model': 'RLinear_OLS',
        'dataset': os.path.basename(data_path).replace('.csv', ''),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'trend_match': trend_match
    }

class Config:
    def __init__(self):
        self.seq_len = 96  # Input sequence length
        self.pred_len = 24  # Prediction length
        self.enc_in = 1  # Number of input features
        self.individual = True  # Use individual linear layers for each channel
        self.batch_size = 32
        self.learning_rate = 0.001

def main():
    # Configuration
    configs = Config()
    
    # Test with a single file first
    data_dir = '/Users/hieuduc/Downloads/rmit/data'
    test_file = 'BTCUSDT.csv'
    data_path = os.path.join(data_dir, test_file)
    
    # Run RLinear OLS experiment
    try:
        result = run_rlinear_ols_experiment(data_path, configs)
        if result is not None:
            print(f"Results for {result['dataset']} - {result['model']}:")
            print(f"  MAE: {result['mae']:.6f}")
            print(f"  MSE: {result['mse']:.6f}")
            print(f"  RMSE: {result['rmse']:.6f}")
            print(f"  Trend Match: {result['trend_match']:.2f}%")
        else:
            print("Experiment failed")
    except Exception as e:
        print(f"Error running RLinear OLS: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
