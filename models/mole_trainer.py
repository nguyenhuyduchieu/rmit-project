"""
Offline RL Training with Off-Policy Evaluation (OPE) for MoLE Router
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

class OfflineRLTrainer:
    """Offline RL trainer with OPE for MoLE router"""
    
    def __init__(self, router, config):
        self.router = router
        self.config = config
        
        # Training parameters
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.rl_learning_rate = config.rl_learning_rate
        self.rl_epochs = config.rl_epochs
        
        # Optimizers
        self.router_optimizer = optim.Adam(
            self.router.parameters(), 
            lr=self.rl_learning_rate
        )
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Training history
        self.training_history = {
            'rewards': [],
            'losses': [],
            'expert_usage': [],
            'ope_estimates': []
        }
    
    def collect_experience(self, states, actions, rewards, next_states, dones):
        """Collect experience for offline RL training"""
        for i in range(len(states)):
            experience = {
                'state': states[i],
                'action': actions[i],
                'reward': rewards[i],
                'next_state': next_states[i],
                'done': dones[i]
            }
            self.experience_buffer.append(experience)
    
    def train_offline(self, logging_policy_data=None):
        """Train router using offline RL with OPE"""
        print("Starting offline RL training...")
        
        if logging_policy_data is not None:
            # Use logged data for training
            self._train_with_logged_data(logging_policy_data)
        else:
            # Use collected experience
            self._train_with_experience()
    
    def _train_with_logged_data(self, logged_data):
        """Train using logged policy data"""
        print("Training with logged policy data...")
        
        for epoch in range(self.rl_epochs):
            epoch_losses = []
            epoch_rewards = []
            
            # Sample batch from logged data
            batch_size = min(32, len(logged_data))
            batch = random.sample(logged_data, batch_size)
            
            for sample in batch:
                state = torch.tensor(sample['state'], dtype=torch.float32)
                logged_action = torch.tensor(sample['logged_action'], dtype=torch.float32)
                reward = sample['reward']
                
                # Get current policy action
                current_action, value = self.router(state.unsqueeze(0))
                current_action = current_action.squeeze(0)
                
                # Calculate importance sampling ratio
                importance_ratio = self._calculate_importance_ratio(
                    current_action, logged_action
                )
                
                # Calculate loss (simplified policy gradient)
                loss = -importance_ratio * reward
                
                # Backward pass
                self.router_optimizer.zero_grad()
                loss.backward()
                self.router_optimizer.step()
                
                epoch_losses.append(loss.item())
                epoch_rewards.append(reward)
            
            # Record training metrics
            avg_loss = np.mean(epoch_losses)
            avg_reward = np.mean(epoch_rewards)
            
            self.training_history['losses'].append(avg_loss)
            self.training_history['rewards'].append(avg_reward)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.6f}, Reward={avg_reward:.6f}")
    
    def _train_with_experience(self):
        """Train using collected experience"""
        print("Training with collected experience...")
        
        if len(self.experience_buffer) < 100:
            print("Not enough experience for training")
            return
        
        for epoch in range(self.rl_epochs):
            epoch_losses = []
            
            # Sample batch from experience buffer
            batch_size = min(32, len(self.experience_buffer))
            batch = random.sample(self.experience_buffer, batch_size)
            
            for experience in batch:
                state = torch.tensor(experience['state'], dtype=torch.float32)
                action = torch.tensor(experience['action'], dtype=torch.float32)
                reward = experience['reward']
                next_state = torch.tensor(experience['next_state'], dtype=torch.float32)
                done = experience['done']
                
                # Get current policy
                current_action, value = self.router(state.unsqueeze(0))
                current_action = current_action.squeeze(0)
                
                # Calculate target value
                with torch.no_grad():
                    _, next_value = self.router(next_state.unsqueeze(0))
                    target_value = reward + self.gamma * next_value.squeeze(0) * (1 - done)
                
                # Calculate loss
                value_loss = nn.MSELoss()(value.squeeze(0), target_value)
                
                # Policy loss (simplified)
                policy_loss = -torch.sum(action * torch.log(current_action + 1e-8))
                
                total_loss = value_loss + 0.1 * policy_loss
                
                # Backward pass
                self.router_optimizer.zero_grad()
                total_loss.backward()
                self.router_optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            avg_loss = np.mean(epoch_losses)
            self.training_history['losses'].append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.6f}")
    
    def _calculate_importance_ratio(self, current_action, logged_action):
        """Calculate importance sampling ratio"""
        # Simplified importance ratio calculation
        # In practice, you'd want more sophisticated methods
        ratio = torch.sum(current_action * logged_action) / (torch.sum(logged_action ** 2) + 1e-8)
        return torch.clamp(ratio, 0.1, 10.0)  # Clipping for stability

class OPEEvaluator:
    """Off-Policy Evaluation for MoLE router"""
    
    def __init__(self):
        self.estimators = ['IPS', 'DR', 'Switch-DR']
    
    def evaluate_policy(self, logged_data, new_policy, estimator='DR'):
        """Evaluate new policy using OPE"""
        
        if estimator == 'IPS':
            return self._ips_estimate(logged_data, new_policy)
        elif estimator == 'DR':
            return self._dr_estimate(logged_data, new_policy)
        elif estimator == 'Switch-DR':
            return self._switch_dr_estimate(logged_data, new_policy)
        else:
            return self._dr_estimate(logged_data, new_policy)
    
    def _ips_estimate(self, logged_data, new_policy):
        """Inverse Propensity Scoring estimate"""
        estimates = []
        
        for sample in logged_data:
            state = torch.tensor(sample['state'], dtype=torch.float32)
            logged_action = sample['logged_action']
            reward = sample['reward']
            
            # Get new policy action
            with torch.no_grad():
                new_action, _ = new_policy(state.unsqueeze(0))
                new_action = new_action.squeeze(0)
            
            # Calculate importance ratio
            importance_ratio = self._calculate_importance_ratio(
                new_action, torch.tensor(logged_action, dtype=torch.float32)
            )
            
            # IPS estimate
            ips_estimate = importance_ratio * reward
            estimates.append(ips_estimate.item())
        
        return np.mean(estimates)
    
    def _dr_estimate(self, logged_data, new_policy):
        """Doubly Robust estimate"""
        estimates = []
        
        for sample in logged_data:
            state = torch.tensor(sample['state'], dtype=torch.float32)
            logged_action = sample['logged_action']
            reward = sample['reward']
            
            # Get new policy action
            with torch.no_grad():
                new_action, value = new_policy(state.unsqueeze(0))
                new_action = new_action.squeeze(0)
            
            # Calculate importance ratio
            importance_ratio = self._calculate_importance_ratio(
                new_action, torch.tensor(logged_action, dtype=torch.float32)
            )
            
            # DR estimate
            dr_estimate = value.item() + importance_ratio * (reward - value.item())
            estimates.append(dr_estimate)
        
        return np.mean(estimates)
    
    def _switch_dr_estimate(self, logged_data, new_policy):
        """Switch-DR estimate (simplified)"""
        # Simplified Switch-DR implementation
        return self._dr_estimate(logged_data, new_policy)
    
    def _calculate_importance_ratio(self, new_action, logged_action):
        """Calculate importance sampling ratio"""
        ratio = torch.sum(new_action * logged_action) / (torch.sum(logged_action ** 2) + 1e-8)
        return torch.clamp(ratio, 0.1, 10.0)

class HeuristicLoggingPolicy:
    """Heuristic logging policy for initial data collection"""
    
    def __init__(self, expert_bank):
        self.expert_bank = expert_bank
    
    def get_action(self, state_features, expert_predictions, expert_uncertainties):
        """Get action using heuristic policy"""
        
        # Strategy 1: Choose expert with lowest uncertainty
        best_expert_idx = np.argmin(expert_uncertainties)
        
        # Strategy 2: Choose expert based on recent performance
        # (simplified - in practice you'd track recent performance)
        
        # Create action vector (one-hot for expert selection)
        action = np.zeros(len(expert_uncertainties))
        action[best_expert_idx] = 1.0
        
        return action
    
    def create_logged_dataset(self, states, expert_predictions, expert_uncertainties, rewards):
        """Create logged dataset using heuristic policy"""
        logged_data = []
        
        for i in range(len(states)):
            # Get action from heuristic policy
            action = self.get_action(
                states[i], 
                expert_predictions[i], 
                expert_uncertainties[i]
            )
            
            logged_sample = {
                'state': states[i],
                'logged_action': action,
                'reward': rewards[i]
            }
            logged_data.append(logged_sample)
        
        return logged_data

class MoLETrainer:
    """Complete trainer for RL-gated MoLE"""
    
    def __init__(self, mole_model, config):
        self.mole_model = mole_model
        self.config = config
        
        # Initialize components
        self.rl_trainer = OfflineRLTrainer(mole_model.router, config)
        self.ope_evaluator = OPEEvaluator()
        self.logging_policy = HeuristicLoggingPolicy(mole_model.expert_bank)
        
        # Training phases
        self.phase = 'expert_training'  # 'expert_training', 'data_collection', 'rl_training'
    
    def train(self, train_data, val_data, test_data):
        """Complete training pipeline"""
        print("="*80)
        print("RL-GATED MOLE TRAINING PIPELINE")
        print("="*80)
        
        # Phase 1: Train experts
        print("\nPhase 1: Training Expert Bank")
        print("-" * 40)
        self.mole_model.train_experts(train_data, val_data)
        
        # Phase 2: Collect logged data
        print("\nPhase 2: Collecting Logged Data")
        print("-" * 40)
        logged_data = self._collect_logged_data(val_data)
        
        # Phase 3: Train RL router
        print("\nPhase 3: Training RL Router")
        print("-" * 40)
        self.rl_trainer.train_offline(logged_data)
        
        # Phase 4: Evaluate with OPE
        print("\nPhase 4: Off-Policy Evaluation")
        print("-" * 40)
        ope_results = self._evaluate_with_ope(logged_data)
        
        return {
            'expert_performance': self.mole_model.get_expert_performance(),
            'rl_training_history': self.rl_trainer.training_history,
            'ope_results': ope_results
        }
    
    def _collect_logged_data(self, val_data):
        """Collect logged data using heuristic policy"""
        print("Collecting logged data...")
        
        logged_data = []
        states = []
        expert_predictions = []
        expert_uncertainties = []
        rewards = []
        
        # Process validation data in batches
        batch_size = 100
        for i in range(0, len(val_data) - self.config.seq_len - self.config.pred_len, batch_size):
            batch_data = val_data[i:i+batch_size]
            
            for j in range(len(batch_data) - self.config.seq_len - self.config.pred_len):
                # Get data window
                window_data = batch_data[j:j+self.config.seq_len]
                target_data = batch_data[j+self.config.seq_len:j+self.config.seq_len+self.config.pred_len]
                
                # Get expert predictions
                X_sample = window_data.reshape(1, -1)
                preds, uncertainties = self.mole_model.expert_bank.predict_all_experts(X_sample)
                
                # Extract state features
                state_feat = self.mole_model.feature_extractor.extract_features(
                    window_data, preds.flatten(), uncertainties
                )
                
                # Calculate reward
                reward = self.mole_model.calculate_reward(
                    torch.tensor(preds.mean(axis=0), dtype=torch.float32),
                    torch.tensor(target_data, dtype=torch.float32)
                )
                
                # Store data
                states.append(state_feat)
                expert_predictions.append(preds.flatten())
                expert_uncertainties.append(uncertainties)
                rewards.append(reward)
        
        # Create logged dataset
        logged_data = self.logging_policy.create_logged_dataset(
            states, expert_predictions, expert_uncertainties, rewards
        )
        
        print(f"Collected {len(logged_data)} logged samples")
        return logged_data
    
    def _evaluate_with_ope(self, logged_data):
        """Evaluate policy using OPE"""
        print("Evaluating policy with OPE...")
        
        ope_results = {}
        
        for estimator in self.ope_evaluator.estimators:
            estimate = self.ope_evaluator.evaluate_policy(
                logged_data, self.mole_model.router, estimator
            )
            ope_results[estimator] = estimate
            print(f"{estimator} estimate: {estimate:.6f}")
        
        return ope_results
