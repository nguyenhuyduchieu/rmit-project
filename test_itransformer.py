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

# Import our iTransformer model
from itransformer_model import iTransformer
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

def run_itransformer_experiment(data_path, configs):
    """Run iTransformer experiment for a single crypto pair"""
    print(f"\n{'='*60}")
    print(f"Running iTransformer on {os.path.basename(data_path)}")
    print(f"{'='*60}")
    
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
    
    # Initialize iTransformer model
    model = iTransformer(configs)
    
    # Train model
    print("Training iTransformer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(5):  # Reduced epochs for testing
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Create dummy x_mark for iTransformer
            x_mark_enc = torch.zeros(batch_x.shape[0], batch_x.shape[1], 0).to(device)
            x_mark_dec = torch.zeros(batch_x.shape[0], batch_y.shape[1], 0).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x, x_mark_enc, batch_y, x_mark_dec)
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
                
                # Create dummy x_mark for iTransformer
                x_mark_enc = torch.zeros(batch_x.shape[0], batch_x.shape[1], 0).to(device)
                x_mark_dec = torch.zeros(batch_x.shape[0], batch_y.shape[1], 0).to(device)
                
                outputs = model(batch_x, x_mark_enc, batch_y, x_mark_dec)
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
    
    # Evaluate model
    print("Evaluating iTransformer...")
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Create dummy x_mark for iTransformer
            x_mark_enc = torch.zeros(batch_x.shape[0], batch_x.shape[1], 0).to(device)
            x_mark_dec = torch.zeros(batch_x.shape[0], batch_y.shape[1], 0).to(device)
            
            outputs = model(batch_x, x_mark_enc, batch_y, x_mark_dec)
            
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
    
    # Concatenate all predictions and true values
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    
    # Inverse transform predictions and true values
    preds_original = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_original = scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)
    
    # Calculate metrics
    mae = MAE(preds_original, trues_original)
    mse = MSE(preds_original, trues_original)
    rmse = RMSE(preds_original, trues_original)
    trend_match = calculate_trend_match(preds_original, trues_original)
    
    return {
        'model': 'iTransformer',
        'dataset': os.path.basename(data_path).replace('.csv', ''),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'trend_match': trend_match
    }

class iTransformerConfig:
    def __init__(self):
        # Basic parameters
        self.seq_len = 96  # Input sequence length
        self.pred_len = 24  # Prediction length
        self.enc_in = 1  # Number of input features
        self.individual = True  # Use individual heads for each channel
        
        # iTransformer specific parameters
        self.d_model = 128  # Model dimension
        self.e_layers = 3  # Number of encoder layers
        self.n_heads = 8  # Number of attention heads
        self.d_ff = 256  # Feed-forward dimension
        self.dropout = 0.1  # Dropout rate
        self.factor = 5  # Attention factor
        self.activation = 'gelu'  # Activation function
        
        # Embedding parameters
        self.embed = 'timeF'  # Embedding type
        self.freq = 'h'  # Frequency
        
        # Model behavior
        self.output_attention = False  # Output attention weights
        self.use_norm = True  # Use normalization
        self.class_strategy = 'last'  # Classification strategy
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.0001

def main():
    # Configuration
    configs = iTransformerConfig()
    
    # Test with a single file first
    data_dir = '/Users/hieuduc/Downloads/rmit/data'
    test_file = 'BTCUSDT.csv'
    data_path = os.path.join(data_dir, test_file)
    
    # Run iTransformer experiment
    try:
        result = run_itransformer_experiment(data_path, configs)
        if result is not None:
            print(f"Results for {result['dataset']} - {result['model']}:")
            print(f"  MAE: {result['mae']:.6f}")
            print(f"  MSE: {result['mse']:.6f}")
            print(f"  RMSE: {result['rmse']:.6f}")
            print(f"  Trend Match: {result['trend_match']:.2f}%")
        else:
            print("Experiment failed")
    except Exception as e:
        print(f"Error running iTransformer: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
