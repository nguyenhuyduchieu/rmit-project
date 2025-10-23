"""
Evaluation metrics for time series forecasting
"""

import numpy as np
from typing import Union

def MAE(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(pred - true))

def MSE(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Squared Error"""
    return np.mean((pred - true) ** 2)

def RMSE(pred: np.ndarray, true: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(MSE(pred, true))

def MAPE(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((pred - true) / (true + 1e-8))) * 100

def SMAPE(pred: np.ndarray, true: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error"""
    return np.mean(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8)) * 100

def calculate_trend_match(pred: np.ndarray, true: np.ndarray) -> float:
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

def calculate_all_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Calculate all evaluation metrics"""
    return {
        'MAE': MAE(pred, true),
        'MSE': MSE(pred, true),
        'RMSE': RMSE(pred, true),
        'MAPE': MAPE(pred, true),
        'SMAPE': SMAPE(pred, true),
        'Trend_Match': calculate_trend_match(pred, true)
    }
