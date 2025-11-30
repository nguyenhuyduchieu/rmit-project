"""
Simplified MoLE model for testing
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SimpleMoLE(nn.Module):
    """Simplified MoLE model for testing"""
    
    def __init__(self, config):
        super(SimpleMoLE, self).__init__()
        
        self.config = config
        self.num_experts = getattr(config, 'num_experts', 4)
        self.seq_len = getattr(config, 'seq_len', 96)
        self.pred_len = getattr(config, 'pred_len', 96)
        
        # Simple experts (just Ridge regression)
        self.experts = nn.ModuleList([
            nn.Linear(self.seq_len, self.pred_len) for _ in range(self.num_experts)
        ])
        
        # Simple router (MLP)
        self.router = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Training history
        self.training_history = {
            'expert_losses': [],
            'router_losses': []
        }
    
    def forward(self, x, return_expert_weights=False):
        """Forward pass"""
        batch_size, seq_len, features = x.shape
        
        # Use only the first feature (Close price)
        x_close = x[:, :, 0]  # [batch_size, seq_len]
        
        # Get expert predictions
        expert_preds = []
        for expert in self.experts:
            pred = expert(x_close)  # [batch_size, pred_len]
            expert_preds.append(pred)
        
        expert_preds = torch.stack(expert_preds, dim=1)  # [batch_size, num_experts, pred_len]
        
        # Get router weights
        router_weights = self.router(x_close)  # [batch_size, num_experts]
        
        # Weighted combination
        weighted_pred = torch.sum(
            router_weights.unsqueeze(-1) * expert_preds, dim=1
        )  # [batch_size, pred_len]
        
        if return_expert_weights:
            return weighted_pred, router_weights
        else:
            return weighted_pred
    
    def train_experts(self, train_loader, val_loader, epochs=10):
        """Train experts using MSE loss"""
        print("Training experts...")
        
        optimizer = torch.optim.Adam(self.experts.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Get expert predictions
                x_close = batch_x[:, :, 0]
                expert_preds = []
                for expert in self.experts:
                    pred = expert(x_close)
                    expert_preds.append(pred)
                
                expert_preds = torch.stack(expert_preds, dim=1)
                
                # Calculate loss for each expert
                expert_losses = []
                for i, expert_pred in enumerate(expert_preds.transpose(0, 1)):
                    loss = criterion(expert_pred, batch_y[:, :, 0])
                    expert_losses.append(loss)
                
                # Average loss across experts
                total_expert_loss = torch.stack(expert_losses).mean()
                total_expert_loss.backward()
                optimizer.step()
                
                total_loss += total_expert_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            self.training_history['expert_losses'].append(avg_loss)
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}: Expert Loss = {avg_loss:.6f}")
    
    def train_router(self, train_loader, val_loader, epochs=10):
        """Train router using expert performance"""
        print("Training router...")
        
        optimizer = torch.optim.Adam(self.router.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Get predictions
                pred, router_weights = self.forward(batch_x, return_expert_weights=True)
                
                # Calculate loss
                loss = criterion(pred, batch_y[:, :, 0])
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            self.training_history['router_losses'].append(avg_loss)
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}: Router Loss = {avg_loss:.6f}")
    
    def train_model(self, train_loader, val_loader, expert_epochs=5, router_epochs=5):
        """Complete training pipeline"""
        print("="*60)
        print("SIMPLE MOLE TRAINING")
        print("="*60)
        
        # Phase 1: Train experts
        self.train_experts(train_loader, val_loader, expert_epochs)
        
        # Phase 2: Train router
        self.train_router(train_loader, val_loader, router_epochs)
        
        print("Training completed!")
    
    def get_expert_performance(self, val_loader):
        """Get performance metrics for each expert"""
        self.train(False)
        criterion = nn.MSELoss()
        
        expert_losses = []
        expert_maes = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x_close = batch_x[:, :, 0]
                
                # Get predictions from each expert
                for i, expert in enumerate(self.experts):
                    pred = expert(x_close)
                    loss = criterion(pred, batch_y[:, :, 0])
                    mae = torch.mean(torch.abs(pred - batch_y[:, :, 0]))
                    
                    if i >= len(expert_losses):
                        expert_losses.append([])
                        expert_maes.append([])
                    
                    expert_losses[i].append(loss.item())
                    expert_maes[i].append(mae.item())
        
        # Calculate average metrics
        performance = []
        for i in range(len(self.experts)):
            avg_loss = np.mean(expert_losses[i])
            avg_mae = np.mean(expert_maes[i])
            performance.append({
                'expert_id': i,
                'val_loss': avg_loss,
                'val_mae': avg_mae
            })
        
        return performance

def test_simple_mole():
    """Test simplified MoLE model"""
    
    print("="*80)
    print("TESTING SIMPLIFIED MOLE MODEL")
    print("="*80)
    
    # Configuration
    data_dir = '/Users/hieuduc/Downloads/rmit/data'
    max_samples = 5000
    
    # Initialize preprocessor
    from src.data.preprocessing import CryptoDataPreprocessor
    from src.data.dataset import create_data_loaders
    from src.configs.model_configs import RLGatedMoLEConfig
    from src.utils.metrics import calculate_all_metrics
    
    preprocessor = CryptoDataPreprocessor(data_dir)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    file_path = os.path.join(data_dir, 'BTCUSDT.csv')
    train_data, valid_data, test_data = preprocessor.process_single_file(
        file_path, max_samples
    )
    
    print(f"Train data: {train_data.shape}")
    print(f"Valid data: {valid_data.shape}")
    print(f"Test data: {test_data.shape}")
    
    # Normalize data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    valid_data_scaled = scaler.transform(valid_data)
    test_data_scaled = scaler.transform(test_data)
    
    # Create data loaders
    config = RLGatedMoLEConfig()
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_data_scaled, valid_data_scaled, test_data_scaled,
        config.seq_len, config.pred_len, config.batch_size
    )
    
    # Initialize Simple MoLE
    mole_model = SimpleMoLE(config)
    
    print(f"\nSimple MoLE Configuration:")
    print(f"  Number of experts: {config.num_experts}")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  Prediction length: {config.pred_len}")
    
    # Train the model
    mole_model.train_model(train_loader, valid_loader, expert_epochs=3, router_epochs=3)
    
    # Get expert performance
    print("\n" + "="*60)
    print("EXPERT PERFORMANCE")
    print("="*60)
    
    expert_performance = mole_model.get_expert_performance(valid_loader)
    for perf in expert_performance:
        print(f"Expert {perf['expert_id']}: Loss={perf['val_loss']:.6f}, MAE={perf['val_mae']:.6f}")
    
    # Test inference
    print("\n" + "="*60)
    print("TESTING INFERENCE")
    print("="*60)
    
    mole_model.eval()
    test_predictions = []
    test_targets = []
    expert_weights_history = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            if i >= 3:  # Test only first 3 batches
                break
            
            # Get predictions
            pred, expert_weights = mole_model(batch_x, return_expert_weights=True)
            
            test_predictions.append(pred.cpu().numpy())
            test_targets.append(batch_y[:, :, 0].cpu().numpy())
            expert_weights_history.append(expert_weights.cpu().numpy())
            
            print(f"Batch {i+1}:")
            print(f"  Prediction shape: {pred.shape}")
            print(f"  Expert weights: {expert_weights[0].numpy()}")
    
    # Calculate metrics
    if test_predictions:
        all_preds = np.concatenate(test_predictions, axis=0)
        all_trues = np.concatenate(test_targets, axis=0)
        
        # Denormalize predictions (simplified)
        # Use only the Close price scaler
        close_scaler = StandardScaler()
        close_scaler.fit(train_data[:, 0:1])
        
        all_preds_denorm = close_scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(all_preds.shape)
        all_trues_denorm = close_scaler.inverse_transform(all_trues.reshape(-1, 1)).reshape(all_trues.shape)
        
        # Calculate metrics
        metrics = calculate_all_metrics(all_preds_denorm, all_trues_denorm)
        
        print(f"\nTest Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        # Analyze expert usage
        all_expert_weights = np.concatenate(expert_weights_history, axis=0)
        avg_expert_usage = np.mean(all_expert_weights, axis=0)
        
        print(f"\nAverage Expert Usage:")
        for i, usage in enumerate(avg_expert_usage):
            print(f"  Expert {i}: {usage:.4f}")
    
    print("\n" + "="*80)
    print("SIMPLE MOLE TESTING COMPLETED")
    print("="*80)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_simple_mole()
