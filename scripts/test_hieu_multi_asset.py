"""
Test HIEU model with multi-asset data
HIEU is designed for multi-asset forecasting, not single asset
"""

import os
import sys
import logging
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from models.HIEU.model import HIEUModel
from models.HIEU.configs import HIEUConfig
from models.HIEU.multi_asset_loader import create_multiasset_loaders
from src.utils.metrics import calculate_all_metrics


def setup_logging():
    """Setup logging"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'test_hieu_multi_asset.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"="*80)
    logger.info(f"Testing HIEU Model with Multi-Asset Data")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"="*80)
    
    return logger


def test_hieu_multi_asset():
    """Test HIEU model with multiple crypto assets"""
    logger = setup_logging()
    
    # Configuration
    data_dir = 'data'
    # Use multiple assets for multi-asset forecasting
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']  # 5 assets
    seq_len = 96
    pred_len = 96
    batch_size = 32
    max_samples = 5000
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Testing with {len(symbols)} assets: {symbols}")
    
    # Create multi-asset data loaders
    try:
        logger.info("Creating multi-asset data loaders...")
        train_loader, valid_loader, test_loader, data_info = create_multiasset_loaders(
            data_dir=data_dir,
            symbols=symbols,
            seq_len=seq_len,
            pred_len=pred_len,
            batch_size=batch_size,
            max_samples=max_samples,
            use_returns=True,  # Use log returns
            log_returns=True,
            standardize=True
        )
        
        logger.info(f"Data info: {data_info.keys()}")
        logger.info(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}, Test batches: {len(test_loader)}")
        
        # Get sample to check dimensions
        sample_x, sample_y = next(iter(train_loader))
        logger.info(f"Input shape: {sample_x.shape} (batch, seq_len, num_assets)")
        logger.info(f"Target shape: {sample_y.shape} (batch, pred_len, num_assets)")
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Initialize HIEU model
    logger.info("Initializing HIEU model...")
    cfg = HIEUConfig()
    cfg.seq_len = seq_len
    cfg.pred_len = pred_len
    cfg.num_nodes = len(symbols)  # Number of assets
    cfg.batch_size = batch_size
    cfg.epochs = 30
    cfg.learning_rate = 1e-3  # Start with higher LR for multi-asset
    cfg.weight_decay = 1e-5
    
    logger.info(f"HIEU Config:")
    logger.info(f"  num_nodes: {cfg.num_nodes}")
    logger.info(f"  seq_len: {cfg.seq_len}")
    logger.info(f"  pred_len: {cfg.pred_len}")
    logger.info(f"  epochs: {cfg.epochs}")
    logger.info(f"  learning_rate: {cfg.learning_rate}")
    
    model = HIEUModel(cfg)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    criterion = torch.nn.MSELoss()
    best_val = float('inf')
    best_state = None
    patience = 10
    bad = 0
    
    # Training loop
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING HIEU MODEL")
    logger.info(f"{'='*60}")
    model.train()
    
    for ep in range(cfg.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device)  # [B, L, N]
            yb = yb.to(device)  # [B, H, N]
            
            optimizer.zero_grad()
            yp = model(xb)  # [B, H, N]
            loss = criterion(yp, yb)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / max(1, num_batches)
        
        # Validation
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xvb, yvb in valid_loader:
                xvb = xvb.to(device)
                yvb = yvb.to(device)
                yvp = model(xvb)
                vloss += criterion(yvp, yvb).item()
        vloss /= max(1, len(valid_loader))
        
        # Update learning rate
        scheduler.step(vloss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log progress
        if (ep + 1) % 5 == 0 or ep == 0:
            logger.info(f"Epoch {ep+1}/{cfg.epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={vloss:.6f}, LR={current_lr:.6f}")
        
        model.train()
        
        # Early stopping
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
            logger.info(f"  ✓ New best validation loss: {best_val:.6f}")
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"  Early stopping at epoch {ep+1}")
                break
    
    # Load best model
    if best_state is not None:
        logger.info(f"\nLoading best model with validation loss: {best_val:.6f}")
        model.load_state_dict(best_state)
    
    # Evaluation
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATING HIEU MODEL")
    logger.info(f"{'='*60}")
    
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yp = model(xb)  # [B, H, N]
            all_preds.append(yp.cpu().numpy())
            all_trues.append(yb.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)  # [n_samples, pred_len, num_assets]
    all_trues = np.concatenate(all_trues, axis=0)  # [n_samples, pred_len, num_assets]
    
    logger.info(f"Predictions shape: {all_preds.shape}")
    logger.info(f"Targets shape: {all_trues.shape}")
    
    # Calculate metrics per asset
    logger.info(f"\n{'='*60}")
    logger.info("METRICS PER ASSET")
    logger.info(f"{'='*60}")
    
    all_metrics = []
    for asset_idx, symbol in enumerate(symbols):
        asset_preds = all_preds[:, :, asset_idx]  # [n_samples, pred_len]
        asset_trues = all_trues[:, :, asset_idx]  # [n_samples, pred_len]
        
        # Reshape for metrics calculation
        asset_preds_flat = asset_preds.reshape(-1, 1)  # [n_samples * pred_len, 1]
        asset_trues_flat = asset_trues.reshape(-1, 1)  # [n_samples * pred_len, 1]
        
        metrics = calculate_all_metrics(asset_preds_flat, asset_trues_flat)
        metrics['asset'] = symbol
        all_metrics.append(metrics)
        
        logger.info(f"\n{symbol}:")
        for k, v in metrics.items():
            if k == 'asset':
                continue
            if k == 'Trend_Match':
                logger.info(f"  {k}: {v:.2f}%")
            else:
                logger.info(f"  {k}: {v:.6f}")
    
    # Calculate average metrics across all assets
    logger.info(f"\n{'='*60}")
    logger.info("AVERAGE METRICS ACROSS ALL ASSETS")
    logger.info(f"{'='*60}")
    
    avg_metrics = {}
    for metric in ['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'Trend_Match']:
        values = [m[metric] for m in all_metrics if metric in m]
        if values:
            avg_metrics[metric] = np.mean(values)
            if metric == 'Trend_Match':
                logger.info(f"  {metric}: {avg_metrics[metric]:.2f}%")
            else:
                logger.info(f"  {metric}: {avg_metrics[metric]:.6f}")
    
    # Save results
    results_df = pd.DataFrame(all_metrics)
    analysis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    results_path = os.path.join(analysis_dir, 'hieu_multi_asset_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")
    
    logger.info(f"\n{'='*80}")
    logger.info("HIEU MULTI-ASSET TEST COMPLETED")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*80}")
    
    return avg_metrics


if __name__ == "__main__":
    test_hieu_multi_asset()

