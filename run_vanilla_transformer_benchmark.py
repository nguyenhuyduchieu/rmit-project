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

# Import our Vanilla Transformer model
from vanilla_transformer_model import VanillaTransformer
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

def load_crypto_data(file_path, target_col='Close', max_samples=100000):
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
    
    # Limit data for faster training if too large
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

def train_vanilla_transformer_model(model, train_loader, valid_loader, configs, epochs=10):
    """Train the Vanilla Transformer model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Create dummy x_mark for Vanilla Transformer
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
                
                # Create dummy x_mark for Vanilla Transformer
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
        
        scheduler.step()
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    return model

def evaluate_vanilla_transformer_model(model, test_loader, scaler, configs):
    """Evaluate the Vanilla Transformer model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Create dummy x_mark for Vanilla Transformer
            x_mark_enc = torch.zeros(batch_x.shape[0], batch_x.shape[1], 0).to(device)
            x_mark_dec = torch.zeros(batch_x.shape[0], batch_y.shape[1], 0).to(device)
            
            outputs = model(batch_x, x_mark_enc, batch_y, x_mark_dec)
            
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

def run_vanilla_transformer_experiment(data_path, configs):
    """Run Vanilla Transformer experiment for a single crypto pair"""
    print(f"\n{'='*60}")
    print(f"Running Vanilla Transformer on {os.path.basename(data_path)}")
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
    
    # Initialize Vanilla Transformer model
    model = VanillaTransformer(configs)
    
    # Train model
    print("Training Vanilla Transformer...")
    trained_model = train_vanilla_transformer_model(model, train_loader, valid_loader, configs)
    
    # Evaluate model
    print("Evaluating Vanilla Transformer...")
    mae, mse, rmse, trend_match = evaluate_vanilla_transformer_model(trained_model, test_loader, scaler, configs)
    
    return {
        'model': 'VanillaTransformer',
        'dataset': os.path.basename(data_path).replace('.csv', ''),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'trend_match': trend_match
    }

class VanillaTransformerConfig:
    def __init__(self):
        # Basic parameters
        self.seq_len = 96  # Input sequence length
        self.pred_len = 24  # Prediction length
        self.enc_in = 1  # Number of input features
        self.dec_in = 1  # Number of decoder input features
        self.c_out = 1  # Number of output features
        self.task_name = 'long_term_forecast'  # Task type
        
        # Vanilla Transformer specific parameters
        self.d_model = 128  # Model dimension
        self.e_layers = 3  # Number of encoder layers
        self.d_layers = 2  # Number of decoder layers
        self.n_heads = 8  # Number of attention heads
        self.d_ff = 256  # Feed-forward dimension
        self.dropout = 0.1  # Dropout rate
        self.factor = 5  # Attention factor
        self.activation = 'gelu'  # Activation function
        
        # Embedding parameters
        self.embed = 'timeF'  # Embedding type
        self.freq = 'h'  # Frequency
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.0001

def main():
    # Configuration
    configs = VanillaTransformerConfig()
    
    # Data directory
    data_dir = '/Users/hieuduc/Downloads/rmit/data'
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    csv_files.sort()
    
    # Results storage
    all_results = []
    
    print(f"Found {len(csv_files)} crypto files to process")
    print("Testing Vanilla Transformer model")
    
    # Run experiments
    for i, csv_file in enumerate(csv_files):
        data_path = os.path.join(data_dir, csv_file)
        print(f"\nProcessing file {i+1}/{len(csv_files)}: {csv_file}")
        
        try:
            result = run_vanilla_transformer_experiment(data_path, configs)
            if result is not None:
                all_results.append(result)
                
                print(f"Results for {result['dataset']} - {result['model']}:")
                print(f"  MAE: {result['mae']:.6f}")
                print(f"  MSE: {result['mse']:.6f}")
                print(f"  RMSE: {result['rmse']:.6f}")
                print(f"  Trend Match: {result['trend_match']:.2f}%")
            
        except Exception as e:
            print(f"Error running Vanilla Transformer on {csv_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('/Users/hieuduc/Downloads/rmit/vanilla_transformer_results.csv', index=False)
        
        # Print summary
        print(f"\n{'='*100}")
        print("VANILLA TRANSFORMER SUMMARY RESULTS")
        print(f"{'='*100}")
        
        # Calculate average metrics
        avg_mae = results_df['mae'].mean()
        avg_mse = results_df['mse'].mean()
        avg_rmse = results_df['rmse'].mean()
        avg_trend_match = results_df['trend_match'].mean()
        
        print(f"\nAverage metrics across all datasets:")
        print(f"MAE: {avg_mae:.6f}")
        print(f"MSE: {avg_mse:.6f}")
        print(f"RMSE: {avg_rmse:.6f}")
        print(f"Trend Match: {avg_trend_match:.2f}%")
        
        # Save summary
        summary_df = pd.DataFrame({
            'model': ['VanillaTransformer'],
            'mae': [avg_mae],
            'mse': [avg_mse],
            'rmse': [avg_rmse],
            'trend_match': [avg_trend_match]
        })
        summary_df.to_csv('/Users/hieuduc/Downloads/rmit/vanilla_transformer_summary.csv', index=False)
        
        # Print detailed results
        print(f"\nDetailed results by dataset:")
        print(results_df.to_string(index=False))
        
        print(f"\nDetailed results saved to: vanilla_transformer_results.csv")
        print(f"Summary results saved to: vanilla_transformer_summary.csv")
    else:
        print("No successful experiments completed.")

if __name__ == "__main__":
    main()
