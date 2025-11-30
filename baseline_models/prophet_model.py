"""
Prophet model wrapper for time series forecasting
Integrates Facebook Prophet into the benchmark framework
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ProphetForecaster:
    """
    Prophet model wrapper that follows the same interface as other models
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.train_df = None
        self.train_timestamps = None
        
    def _prepare_prophet_data(self, data: np.ndarray, timestamps: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """
        Convert numpy array to Prophet format (DataFrame with 'ds' and 'y' columns)
        
        Args:
            data: numpy array of shape (n_samples, n_features) or (n_samples,)
            timestamps: optional datetime index
            
        Returns:
            DataFrame with 'ds' (datetime) and 'y' (target value) columns
        """
        # Extract the target column (usually first column which is Close price)
        if len(data.shape) > 1:
            y = data[:, 0]  # Use first feature as target
        else:
            y = data
            
        # Create timestamps if not provided
        if timestamps is None:
            # Generate hourly timestamps starting from a base date
            n_samples = len(y)
            timestamps = pd.date_range(start='2020-01-01', periods=n_samples, freq='15T')
        
        # Create Prophet DataFrame
        prophet_df = pd.DataFrame({
            'ds': timestamps[:len(y)],
            'y': y
        })
        
        return prophet_df
    
    def _create_timestamps(self, start_date: str, n_samples: int, freq: str = '15T') -> pd.DatetimeIndex:
        """Create datetime index for time series"""
        return pd.date_range(start=start_date, periods=n_samples, freq=freq)
    
    def train(self, train_data: np.ndarray, valid_data: Optional[np.ndarray] = None):
        """
        Train Prophet model
        
        Args:
            train_data: numpy array of shape (n_samples, n_features)
            valid_data: optional validation data (not used by Prophet but kept for consistency)
        """
        # Prepare training data for Prophet
        # Prophet needs full time series, not sliding windows
        # We'll use the raw time series values
        
        # Extract target variable (first column)
        if len(train_data.shape) > 1:
            train_values = train_data[:, 0]
        else:
            train_values = train_data
        
        # Create timestamps
        self.train_timestamps = self._create_timestamps('2020-01-01', len(train_values))
        
        # Prepare Prophet DataFrame
        self.train_df = self._prepare_prophet_data(train_data, self.train_timestamps)
        
        # Initialize and train Prophet
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,  # We use 15-min data, daily seasonality might be too granular
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        # Fit the model
        print(f"Training Prophet model on {len(self.train_df)} samples...")
        self.model.fit(self.train_df)
        print("Prophet model trained successfully")
    
    def predict(self, test_data: np.ndarray, test_timestamps: Optional[pd.DatetimeIndex] = None) -> np.ndarray:
        """
        Make predictions for test data
        
        Args:
            test_data: numpy array of shape (n_samples, n_features) or (n_samples,)
            test_timestamps: optional datetime index for test data
            
        Returns:
            Predictions as numpy array matching test_data format
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract test target values
        if len(test_data.shape) > 1:
            test_values = test_data[:, 0]
            n_features = test_data.shape[1]
        else:
            test_values = test_data
            n_features = 1
        
        # Create future dataframe for predictions
        n_test = len(test_values)
        
        # Create future timestamps
        if test_timestamps is None:
            # Continue from training data timestamps
            if self.train_timestamps is not None:
                last_train_time = self.train_timestamps[-1]
                future_timestamps = pd.date_range(
                    start=last_train_time + pd.Timedelta(minutes=15),
                    periods=n_test,
                    freq='15T'
                )
            else:
                future_timestamps = self._create_timestamps('2025-01-01', n_test)
        else:
            future_timestamps = test_timestamps[:n_test]
        
        future_df = pd.DataFrame({'ds': future_timestamps})
        
        # Make predictions
        forecast = self.model.predict(future_df)
        predictions = forecast['yhat'].values
        
        # Reshape to match test_data format
        if len(test_data.shape) > 1:
            # If test_data has multiple features, return predictions in same shape
            predictions = predictions.reshape(-1, 1)
            # If test_data has more than 1 feature, pad with zeros or repeat
            if n_features > 1:
                predictions = np.tile(predictions, (1, n_features))
        
        return predictions
    
    def evaluate(self, test_data: np.ndarray, test_timestamps: Optional[pd.DatetimeIndex] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate model on test data
        
        Args:
            test_data: numpy array of shape (n_samples, n_features)
            test_timestamps: optional datetime index for test data
            
        Returns:
            Tuple of (predictions, true_values)
        """
        predictions = self.predict(test_data, test_timestamps)
        
        # Extract true values
        if len(test_data.shape) > 1:
            true_values = test_data[:, 0:1]  # Match prediction shape
        else:
            true_values = test_data
        
        return predictions, true_values


class ProphetModel:
    """
    Prophet model wrapper compatible with the benchmark framework
    Works with sliding window data format by training on full series
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_full_data = None
        self.train_full_timestamps = None
        self.last_train_index = 0
        
    def train_on_full_series(self, train_data: np.ndarray, valid_data: Optional[np.ndarray] = None):
        """
        Train on full time series (Prophet doesn't use sliding windows)
        
        Args:
            train_data: full time series array (n_samples, n_features)
            valid_data: optional validation data (combined with train for Prophet)
        """
        self.train_full_data = train_data
        
        # Combine train and valid for Prophet (Prophet works better with more data)
        if valid_data is not None:
            combined_data = np.vstack([train_data, valid_data])
        else:
            combined_data = train_data
        
        # Extract target (first column)
        if len(combined_data.shape) > 1:
            train_values = combined_data[:, 0]
        else:
            train_values = combined_data
        
        self.last_train_index = len(train_values)
        
        # Create timestamps
        self.train_full_timestamps = pd.date_range(
            start='2020-01-01',
            periods=len(train_values),
            freq='15T'
        )
        
        # Prepare Prophet DataFrame
        prophet_train_df = pd.DataFrame({
            'ds': self.train_full_timestamps,
            'y': train_values
        })
        
        # Initialize Prophet with config parameters
        self.model = Prophet(
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=self.config.daily_seasonality,
            seasonality_mode=self.config.seasonality_mode,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            interval_width=self.config.interval_width
        )
        
        print(f"Training Prophet on {len(prophet_train_df)} samples...")
        self.model.fit(prophet_train_df)
        print("Prophet training completed")
    
    def predict_sliding_windows(self, test_loader):
        """
        Make predictions for sliding window test data
        
        Args:
            test_loader: DataLoader with batches of (batch_x, batch_y)
            
        Returns:
            Tuple of (all_predictions, all_targets) as numpy arrays
        """
        import torch
        
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        all_predictions = []
        all_targets = []
        
        # Collect all test windows first to understand the structure
        test_batches = []
        for batch_x, batch_y in test_loader:
            test_batches.append((batch_x.numpy(), batch_y.numpy()))
        
        # Prophet needs continuous predictions from the training end
        # We'll predict all future points at once, then slice for each window
        
        # Calculate how many steps ahead we need to predict
        if test_batches:
            seq_len = test_batches[0][0].shape[1]
            pred_len = test_batches[0][1].shape[1]
            n_test_windows = sum(len(batch_x) for batch_x, _ in test_batches)
            
            # Total prediction length needed
            total_pred_steps = n_test_windows + pred_len - 1
            
            # Create future dataframe from end of training
            last_train_time = self.train_full_timestamps[-1]
            future_dates = pd.date_range(
                start=last_train_time + pd.Timedelta(minutes=15),
                periods=total_pred_steps,
                freq='15T'
            )
            
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Make predictions for all future points
            forecast = self.model.predict(future_df)
            all_future_preds = forecast['yhat'].values  # (total_pred_steps,)
            
            # Now extract predictions for each sliding window
            window_idx = 0
            for batch_x, batch_y in test_batches:
                batch_predictions = []
                
                for i in range(len(batch_x)):
                    # Each window represents predictions starting at different time points
                    # Window 0: predictions[0:pred_len]
                    # Window 1: predictions[1:1+pred_len]
                    # etc.
                    start_idx = window_idx
                    end_idx = start_idx + pred_len
                    
                    if end_idx <= len(all_future_preds):
                        window_preds = all_future_preds[start_idx:end_idx]
                    else:
                        # If we run out, use the last predictions
                        window_preds = all_future_preds[start_idx:]
                        # Pad if necessary
                        if len(window_preds) < pred_len:
                            padding = np.full(pred_len - len(window_preds), all_future_preds[-1])
                            window_preds = np.concatenate([window_preds, padding])
                    
                    batch_predictions.append(window_preds)
                    window_idx += 1  # Move to next window
                
                batch_preds = np.array(batch_predictions)
                all_predictions.append(batch_preds)
                all_targets.append(batch_y[:, :, 0])  # Extract target from batch_y
        
        # Concatenate all batches
        if all_predictions:
            all_preds = np.concatenate(all_predictions, axis=0)
            all_trues = np.concatenate(all_targets, axis=0)
            
            # Reshape to match expected format: (n_samples, pred_len, 1)
            all_preds = all_preds.reshape(len(all_preds), -1, 1)
            all_trues = all_trues.reshape(len(all_trues), -1, 1)
            
            return all_preds, all_trues
        else:
            raise ValueError("No test batches found")

