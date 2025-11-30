"""
RL-gated Mixture-of-RLinear Experts (MoLE) Implementation
Based on MoLE (AISTATS'24) with RL router for utility optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

from models.revin import RevIN
from baseline_models.linear_models import RLinearOLS

class RLinearExpert:
    """Individual RLinear expert with specific configuration"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.use_revin = config.get('use_revin', True)
        self.target_type = config.get('target_type', 'levels')  # 'levels' or 'returns'
        self.coin_cluster = config.get('coin_cluster', 'BTC-led')
        
        # Initialize RevIN if needed
        self.revin = RevIN(config['enc_in']) if self.use_revin else None
        
        # Initialize RLinear model (will be properly initialized in fit method)
        self.model = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the expert"""
        # Create a simple linear model instead of RLinearOLS for now
        from sklearn.linear_model import Ridge
        
        # Flatten data for training
        X_flat = X_train.reshape(X_train.shape[0], -1)
        y_flat = y_train.reshape(y_train.shape[0], -1)
        
        # Train Ridge regression
        self.model = Ridge(alpha=0.000001)
        self.model.fit(X_flat, y_flat)
        
        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            val_pred = self.model.predict(X_val_flat)
            y_val_flat = y_val.reshape(y_val.shape[0], -1)
            val_loss = np.mean((val_pred - y_val_flat) ** 2)
            self.val_losses.append(val_loss)
            
            # Calculate additional metrics
            mae = np.mean(np.abs(val_pred - y_val_flat))
            mda = self._calculate_mda(val_pred, y_val_flat)
            self.val_metrics.append({'mae': mae, 'mda': mda})
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Flatten input for prediction
        X_flat = X.reshape(X.shape[0], -1)
        pred = self.model.predict(X_flat)
        
        # Reshape back to original shape
        return pred.reshape(X.shape[0], self.config['pred_len'])
    
    def _calculate_mda(self, pred, true):
        """Calculate Mean Directional Accuracy"""
        pred_direction = np.sign(np.diff(pred.flatten()))
        true_direction = np.sign(np.diff(true.flatten()))
        return np.mean(pred_direction == true_direction) * 100

class ExpertBank:
    """Bank of RLinear experts with different configurations"""
    
    def __init__(self, num_experts: int = 8):
        self.num_experts = num_experts
        self.experts = []
        self.expert_configs = self._create_expert_configs()
        
    def _create_expert_configs(self) -> List[Dict]:
        """Create diverse expert configurations"""
        configs = []
        
        # Different lookback windows
        lookbacks = [96, 192, 384]
        
        # Different patch sizes
        patches = [8, 16]
        
        # RevIN on/off
        revin_options = [True, False]
        
        # Target types
        target_types = ['levels', 'returns']
        
        # Coin clusters (simplified)
        clusters = ['BTC-led', 'L1', 'DeFi', 'Stable']
        
        config_id = 0
        for lookback in lookbacks:
            for patch in patches:
                for use_revin in revin_options:
                    for target_type in target_types:
                        if config_id >= self.num_experts:
                            break
                        
                        config = {
                            'seq_len': lookback,
                            'pred_len': 96,  # Standard prediction length
                            'enc_in': 1,  # Single target variable
                            'use_revin': use_revin,
                            'target_type': target_type,
                            'coin_cluster': clusters[config_id % len(clusters)],
                            'patch_len': patch,
                            'stride': patch // 2,
                            'expert_id': config_id
                        }
                        configs.append(config)
                        config_id += 1
                        
                    if config_id >= self.num_experts:
                        break
                if config_id >= self.num_experts:
                    break
            if config_id >= self.num_experts:
                break
                
        return configs[:self.num_experts]
    
    def initialize_experts(self):
        """Initialize all experts"""
        self.experts = []
        for config in self.expert_configs:
            expert = RLinearExpert(config)
            self.experts.append(expert)
    
    def train_experts(self, train_data, val_data):
        """Train all experts"""
        print(f"Training {len(self.experts)} experts...")
        
        for i, expert in enumerate(self.experts):
            print(f"Training expert {i+1}/{len(self.experts)}: {expert.config}")
            
            # Prepare data for this expert
            X_train, y_train = self._prepare_expert_data(train_data, expert.config)
            X_val, y_val = self._prepare_expert_data(val_data, expert.config)
            
            # Train expert
            expert.fit(X_train, y_train, X_val, y_val)
            
            print(f"  Val Loss: {expert.val_losses[-1]:.6f}")
            if expert.val_metrics:
                print(f"  Val MAE: {expert.val_metrics[-1]['mae']:.6f}")
                print(f"  Val MDA: {expert.val_metrics[-1]['mda']:.2f}%")
    
    def _prepare_expert_data(self, data, config):
        """Prepare data for specific expert configuration"""
        seq_len = config['seq_len']
        pred_len = config['pred_len']
        
        # Create sliding windows
        X, y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len:i+seq_len+pred_len])
        
        return np.array(X), np.array(y)
    
    def predict_all_experts(self, X):
        """Get predictions from all experts"""
        predictions = []
        uncertainties = []
        
        for expert in self.experts:
            pred = expert.predict(X)
            predictions.append(pred)
            
            # Simple uncertainty estimation based on validation performance
            if expert.val_losses:
                uncertainty = np.sqrt(expert.val_losses[-1])
            else:
                uncertainty = 1.0
            uncertainties.append(uncertainty)
        
        return np.array(predictions), np.array(uncertainties)

class RLRouter(nn.Module):
    """RL-based router for expert selection"""
    
    def __init__(self, state_dim: int, num_experts: int, hidden_dim: int = 256):
        super(RLRouter, self).__init__()
        
        self.state_dim = state_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Expert selection head (softmax weights)
        self.expert_head = nn.Linear(hidden_dim, num_experts)
        
        # Value head for critic
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """Forward pass"""
        # Encode state
        encoded = self.state_encoder(state)
        
        # Get expert weights
        expert_logits = self.expert_head(encoded)
        expert_weights = F.softmax(expert_logits, dim=-1)
        
        # Get value estimate
        value = self.value_head(encoded)
        
        return expert_weights, value
    
    def get_action(self, state):
        """Get expert selection action"""
        with torch.no_grad():
            expert_weights, _ = self.forward(state)
            return expert_weights

class StateFeatureExtractor:
    """Extract state features for RL router"""
    
    def __init__(self, lookback_window: int = 24):
        self.lookback_window = lookback_window
        
    def extract_features(self, data, expert_predictions=None, expert_uncertainties=None):
        """Extract comprehensive state features"""
        features = []
        
        # 1. Volatility features (1h-1d)
        vol_1h = self._calculate_volatility(data, window=4)  # 4 * 15min = 1h
        vol_4h = self._calculate_volatility(data, window=16)  # 16 * 15min = 4h
        vol_1d = self._calculate_volatility(data, window=96)  # 96 * 15min = 1d
        
        features.extend([vol_1h, vol_4h, vol_1d])
        
        # 2. Regime detection (simplified CPD)
        regime_flags = self._detect_regime_change(data)
        features.extend(regime_flags)
        
        # 3. Recent performance metrics
        if expert_predictions is not None and expert_uncertainties is not None:
            # Expert performance features
            best_expert_idx = np.argmin(expert_uncertainties)
            worst_expert_idx = np.argmax(expert_uncertainties)
            
            features.extend([
                expert_uncertainties[best_expert_idx],
                expert_uncertainties[worst_expert_idx],
                np.mean(expert_uncertainties),
                np.std(expert_uncertainties)
            ])
            
            # Prediction diversity
            pred_std = np.std(expert_predictions, axis=0).mean()
            features.append(pred_std)
        else:
            # Default values when no expert predictions available
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 4. Liquidity proxy (simplified)
        liquidity_proxy = self._calculate_liquidity_proxy(data)
        features.append(liquidity_proxy)
        
        # 5. Trend features
        trend_features = self._calculate_trend_features(data)
        features.extend(trend_features)
        
        return np.array(features)
    
    def _calculate_volatility(self, data, window):
        """Calculate rolling volatility"""
        if len(data) < window:
            return 0.0
        
        returns = np.diff(data[-window:])
        return np.std(returns) if len(returns) > 0 else 0.0
    
    def _detect_regime_change(self, data):
        """Detect regime changes (simplified)"""
        if len(data) < 20:
            return [0.0, 0.0, 0.0]  # No regime change detected
        
        # Simple regime detection based on volatility and trend
        recent_vol = self._calculate_volatility(data, window=10)
        older_vol = self._calculate_volatility(data[:-10], window=10) if len(data) > 20 else recent_vol
        
        vol_change = abs(recent_vol - older_vol) / (older_vol + 1e-8)
        
        # Trend change detection
        recent_trend = np.mean(np.diff(data[-10:]))
        older_trend = np.mean(np.diff(data[-20:-10])) if len(data) > 20 else recent_trend
        
        trend_change = abs(recent_trend - older_trend)
        
        # Market state (bull/bear/sideways)
        overall_trend = np.mean(np.diff(data[-20:])) if len(data) > 20 else 0.0
        market_state = 1.0 if overall_trend > 0 else (-1.0 if overall_trend < 0 else 0.0)
        
        return [vol_change, trend_change, market_state]
    
    def _calculate_liquidity_proxy(self, data):
        """Calculate liquidity proxy (simplified)"""
        if len(data) < 10:
            return 0.0
        
        # Use price stability as liquidity proxy
        recent_prices = data[-10:]
        price_stability = 1.0 / (np.std(recent_prices) + 1e-8)
        return min(price_stability, 10.0)  # Cap at 10
    
    def _calculate_trend_features(self, data):
        """Calculate trend-related features"""
        if len(data) < 20:
            return [0.0, 0.0, 0.0]
        
        # Short-term trend
        short_trend = np.mean(np.diff(data[-5:]))
        
        # Medium-term trend  
        medium_trend = np.mean(np.diff(data[-10:]))
        
        # Long-term trend
        long_trend = np.mean(np.diff(data[-20:]))
        
        return [short_trend, medium_trend, long_trend]

class RewardCalculator:
    """Calculate rewards for RL training"""
    
    def __init__(self, reward_type: str = 'mda_rmse', beta: float = 0.5):
        self.reward_type = reward_type
        self.beta = beta
        
    def calculate_reward(self, predictions, true_values, expert_weights=None):
        """Calculate reward based on predictions"""
        
        if self.reward_type == 'mda_rmse':
            return self._mda_rmse_reward(predictions, true_values)
        elif self.reward_type == 'pnl_cvar':
            return self._pnl_cvar_reward(predictions, true_values)
        else:
            return self._mda_rmse_reward(predictions, true_values)
    
    def _mda_rmse_reward(self, predictions, true_values):
        """Calculate MDA + RMSE reward"""
        # Calculate RMSE
        rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
        
        # Calculate MDA
        pred_direction = np.sign(np.diff(predictions.flatten()))
        true_direction = np.sign(np.diff(true_values.flatten()))
        mda = np.mean(pred_direction == true_direction) * 100
        
        # Combined reward: higher MDA, lower RMSE
        reward = mda - self.beta * rmse
        
        return reward
    
    def _pnl_cvar_reward(self, predictions, true_values):
        """Calculate PnL-based reward with CVaR penalty"""
        # Simple trading strategy: long if prediction > 0, short if < 0
        positions = np.sign(predictions)
        returns = np.diff(true_values.flatten())
        
        # Calculate PnL
        pnl = np.sum(positions[:-1] * returns)
        
        # Calculate CVaR (simplified)
        if len(returns) > 0:
            sorted_returns = np.sort(returns)
            var_5pct = np.percentile(sorted_returns, 5)
            cvar = np.mean(sorted_returns[sorted_returns <= var_5pct])
        else:
            cvar = 0.0
        
        # Combined reward: higher PnL, lower CVaR
        reward = pnl - self.beta * abs(cvar)
        
        return reward

class RLGatedMoLE(nn.Module):
    """RL-gated Mixture-of-RLinear Experts"""
    
    def __init__(self, config):
        super(RLGatedMoLE, self).__init__()
        
        self.config = config
        self.num_experts = getattr(config, 'num_experts', 8)
        self.state_dim = getattr(config, 'state_dim', 20)  # Feature dimension
        
        # Initialize components
        self.expert_bank = ExpertBank(self.num_experts)
        self.expert_bank.initialize_experts()
        
        self.router = RLRouter(
            state_dim=self.state_dim,
            num_experts=self.num_experts,
            hidden_dim=getattr(config, 'router_hidden_dim', 256)
        )
        
        self.feature_extractor = StateFeatureExtractor()
        self.reward_calculator = RewardCalculator(
            reward_type=getattr(config, 'reward_type', 'mda_rmse'),
            beta=getattr(config, 'reward_beta', 0.5)
        )
        
        # Training history
        self.training_history = {
            'rewards': [],
            'expert_weights': [],
            'state_features': []
        }
    
    def forward(self, x, return_expert_weights=False):
        """Forward pass"""
        batch_size, seq_len, features = x.shape
        
        # Extract state features
        state_features = []
        expert_predictions = []
        
        for i in range(batch_size):
            # Get data for this sample
            sample_data = x[i, :, 0].cpu().numpy()  # Assuming first feature is target
            
            # Get predictions from all experts
            X_sample = sample_data.reshape(1, -1)
            preds, uncertainties = self.expert_bank.predict_all_experts(X_sample)
            expert_predictions.append(preds)
            
            # Extract state features
            state_feat = self.feature_extractor.extract_features(
                sample_data, preds.flatten(), uncertainties
            )
            state_features.append(state_feat)
        
        # Convert to tensors
        state_features = torch.tensor(np.array(state_features), dtype=torch.float32)
        expert_predictions = torch.tensor(np.array(expert_predictions), dtype=torch.float32)
        
        # Get expert weights from router
        expert_weights, value = self.router(state_features)
        
        # Weighted combination of expert predictions
        weighted_predictions = torch.sum(
            expert_weights.unsqueeze(-1) * expert_predictions, dim=1
        )
        
        if return_expert_weights:
            return weighted_predictions, expert_weights, value
        else:
            return weighted_predictions
    
    def train_experts(self, train_data, val_data):
        """Train all experts"""
        self.expert_bank.train_experts(train_data, val_data)
    
    def calculate_reward(self, predictions, true_values, expert_weights=None):
        """Calculate reward for RL training"""
        return self.reward_calculator.calculate_reward(
            predictions.cpu().numpy(), 
            true_values.cpu().numpy(), 
            expert_weights
        )
    
    def get_expert_performance(self):
        """Get performance metrics for all experts"""
        performance = []
        for i, expert in enumerate(self.expert_bank.experts):
            if expert.val_metrics:
                perf = {
                    'expert_id': i,
                    'config': expert.config,
                    'val_loss': expert.val_losses[-1] if expert.val_losses else float('inf'),
                    'val_mae': expert.val_metrics[-1]['mae'],
                    'val_mda': expert.val_metrics[-1]['mda']
                }
                performance.append(perf)
        return performance
