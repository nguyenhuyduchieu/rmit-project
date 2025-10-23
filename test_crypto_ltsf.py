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

# Import LTSF-Linear models
import sys
sys.path.append('/Users/hieuduc/Downloads/rmit/paper_1/LTSF-Linear')

from models.Linear import Model as LinearModel
from models.DLinear import Model as DLinearModel
from models.NLinear import Model as NLinearModel
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
    
    return trend_match

def train_model(model, train_loader, valid_loader, configs, epochs=5):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
        
        print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    return model

def evaluate_model(model, test_loader, scaler, configs):
    """Evaluate the model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
    
    # Concatenate all predictions and true values
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    
    # Inverse transform if scaler is provided
    if scaler is not None:
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
        trues = scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)
    
    # Calculate metrics
    mae = MAE(preds, trues)
    mse = MSE(preds, trues)
    rmse = RMSE(preds, trues)
    trend_match = calculate_trend_match(preds, trues)
    
    return mae, mse, rmse, trend_match

def run_experiment(data_path, model_name, configs):
    """Run experiment for a single crypto pair"""
    print(f"\n{'='*50}")
    print(f"Running {model_name} on {os.path.basename(data_path)}")
    print(f"{'='*50}")
    
    # Load data
    train_data, valid_data, test_data = load_crypto_data(data_path)
    
    # Check if we have enough data
    if len(train_data) < configs.seq_len + configs.pred_len:
        print(f"Not enough training data: {len(train_data)} samples")
        return None
    
    if len(valid_data) < configs.seq_len + configs.pred_len:
        print(f"Not enough validation data: {len(valid_data)} samples")
        return None
        
    if len(test_data) < configs.seq_len + configs.pred_len:
        print(f"Not enough test data: {len(test_data)} samples")
        return None
    
    # Normalize data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    valid_data_scaled = scaler.transform(valid_data)
    test_data_scaled = scaler.transform(test_data)
    
    # Create datasets
    train_dataset = CryptoDataset(train_data_scaled, configs.seq_len, configs.pred_len)
    valid_dataset = CryptoDataset(valid_data_scaled, configs.seq_len, configs.pred_len)
    test_dataset = CryptoDataset(test_data_scaled, configs.seq_len, configs.pred_len)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=configs.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False)
    
    # Initialize model
    if model_name == 'Linear':
        model = LinearModel(configs)
    elif model_name == 'DLinear':
        model = DLinearModel(configs)
    elif model_name == 'NLinear':
        model = NLinearModel(configs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train model
    print(f"Training {model_name}...")
    trained_model = train_model(model, train_loader, valid_loader, configs)
    
    # Evaluate model
    print(f"Evaluating {model_name}...")
    mae, mse, rmse, trend_match = evaluate_model(trained_model, test_loader, scaler, configs)
    
    return {
        'model': model_name,
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
    
    # Models to test
    models = ['Linear', 'DLinear', 'NLinear']
    
    # Results storage
    all_results = []
    
    # Run experiments
    for model_name in models:
        try:
            result = run_experiment(data_path, model_name, configs)
            if result is not None:
                all_results.append(result)
                
                print(f"Results for {result['dataset']} - {result['model']}:")
                print(f"  MAE: {result['mae']:.6f}")
                print(f"  MSE: {result['mse']:.6f}")
                print(f"  RMSE: {result['rmse']:.6f}")
                print(f"  Trend Match: {result['trend_match']:.2f}%")
                
        except Exception as e:
            print(f"Error running {model_name} on {test_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    if all_results:
        print(f"\n{'='*80}")
        print("SUMMARY RESULTS")
        print(f"{'='*80}")
        
        for result in all_results:
            print(f"{result['model']:10} - MAE: {result['mae']:.6f}, MSE: {result['mse']:.6f}, RMSE: {result['rmse']:.6f}, Trend Match: {result['trend_match']:.2f}%")

if __name__ == "__main__":
    main()
