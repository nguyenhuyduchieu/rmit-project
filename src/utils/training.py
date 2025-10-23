"""
Training utilities for time series models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time

class ModelTrainer:
    """Generic trainer for time series forecasting models"""
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        self.model = model
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device)
        
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, model_args: Dict[str, Any] = None) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            # Prepare model arguments
            if model_args is None:
                model_args = {}
            
            # Add dummy temporal features if needed
            if 'x_mark_enc' not in model_args:
                model_args['x_mark_enc'] = torch.zeros(batch_x.shape[0], batch_x.shape[1], 0).to(self.device)
            if 'x_mark_dec' not in model_args:
                model_args['x_mark_dec'] = torch.zeros(batch_x.shape[0], batch_y.shape[1], 0).to(self.device)
            
            # Forward pass - handle different model types
            if hasattr(self.model, 'forward'):
                # Check if model expects additional arguments
                import inspect
                sig = inspect.signature(self.model.forward)
                if 'x_mark_enc' in sig.parameters:
                    outputs = self.model(batch_x, **model_args)
                else:
                    outputs = self.model(batch_x)
            else:
                outputs = self.model(batch_x, batch_y, **model_args)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, valid_loader: DataLoader, criterion: nn.Module, 
                      model_args: Dict[str, Any] = None) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Prepare model arguments
                if model_args is None:
                    model_args = {}
                
                # Add dummy temporal features if needed
                if 'x_mark_enc' not in model_args:
                    model_args['x_mark_enc'] = torch.zeros(batch_x.shape[0], batch_x.shape[1], 0).to(self.device)
                if 'x_mark_dec' not in model_args:
                    model_args['x_mark_dec'] = torch.zeros(batch_x.shape[0], batch_y.shape[1], 0).to(self.device)
                
                # Forward pass - handle different model types
                if hasattr(self.model, 'forward'):
                    # Check if model expects additional arguments
                    import inspect
                    sig = inspect.signature(self.model.forward)
                    if 'x_mark_enc' in sig.parameters:
                        outputs = self.model(batch_x, **model_args)
                    else:
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, batch_y, **model_args)
                
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(valid_loader)
    
    def train(self, train_loader: DataLoader, valid_loader: DataLoader, 
              epochs: int = 10, learning_rate: float = 0.001, 
              scheduler_type: str = 'cosine', model_args: Dict[str, Any] = None,
              verbose: bool = True) -> Dict[str, Any]:
        """Train the model with early stopping"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Setup scheduler
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
        else:
            scheduler = None
        
        best_val_loss = float('inf')
        best_model_state = None
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion, model_args)
            
            # Validate
            val_loss = self.validate_epoch(valid_loader, criterion, model_args)
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            epoch_time = time.time() - start_time
            
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f'Epoch {epoch:3d}/{epochs}: Train Loss: {train_loss:.6f}, '
                      f'Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s')
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
    
    def evaluate(self, test_loader: DataLoader, scaler: Optional[StandardScaler] = None,
                model_args: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the model on test set"""
        self.model.eval()
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Prepare model arguments
                if model_args is None:
                    model_args = {}
                
                # Add dummy temporal features if needed
                if 'x_mark_enc' not in model_args:
                    model_args['x_mark_enc'] = torch.zeros(batch_x.shape[0], batch_x.shape[1], 0).to(self.device)
                if 'x_mark_dec' not in model_args:
                    model_args['x_mark_dec'] = torch.zeros(batch_x.shape[0], batch_y.shape[1], 0).to(self.device)
                
                # Forward pass - handle different model types
                if hasattr(self.model, 'forward'):
                    # Check if model expects additional arguments
                    import inspect
                    sig = inspect.signature(self.model.forward)
                    if 'x_mark_enc' in sig.parameters:
                        outputs = self.model(batch_x, **model_args)
                    else:
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, batch_y, **model_args)
                
                all_preds.append(outputs.cpu().numpy())
                all_trues.append(batch_y.cpu().numpy())
        
        # Concatenate all predictions and true values
        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        
        # Inverse transform if scaler is provided
        if scaler is not None:
            preds = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
            trues = scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)
        
        return preds, trues
