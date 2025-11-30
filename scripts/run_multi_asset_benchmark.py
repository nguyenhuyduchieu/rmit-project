"""
Multi-Asset Benchmark Script
Run all models on multi-asset data (5 cryptocurrencies)
"""

import os
import sys
import logging
from datetime import datetime
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from models.HIEU.multi_asset_loader import create_multiasset_loaders
from src.utils.metrics import calculate_all_metrics

# Import models
from baseline_models.linear_models import Linear, DLinear, NLinear
from baseline_models.rlinear_model import RLinearModel
from baseline_models.itransformer_model import iTransformer
from baseline_models.patchtst_model import PatchTST
from baseline_models.transformer_models import VanillaTransformer
from models.HIEU.model import HIEUModel
from models.HIEU.configs import HIEUConfig
from scripts.test_simple_mole import SimpleMoLE

# Import configs
from src.configs.model_configs import (
    LinearConfig, DLinearConfig, NLinearConfig, RLinearConfig,
    PatchTSTConfig, iTransformerConfig, VanillaTransformerConfig, RLGatedMoLEConfig
)


def setup_logging():
    """Setup logging"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'multi_asset_benchmark.log')
    
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
    logger.info(f"Multi-Asset Benchmark - All Models")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"="*80)
    
    return logger


def train_and_evaluate_model(model, train_loader, valid_loader, test_loader, 
                             model_name, device, logger, epochs=20):
    """Train and evaluate a model on multi-asset data"""
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False, min_lr=1e-6
    )
    criterion = torch.nn.MSELoss()
    
    best_val = float('inf')
    best_state = None
    patience = 10
    bad = 0
    
    # Training
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")
    
    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device)  # [B, L, N]
            yb = yb.to(device)  # [B, H, N]
            
            optimizer.zero_grad()
            yp = model(xb)  # [B, H, N]
            
            # Ensure output shape matches target
            if yp.shape != yb.shape:
                if len(yp.shape) == 2:
                    yp = yp.unsqueeze(-1).expand(-1, -1, yb.shape[-1])
                elif yp.shape[-1] != yb.shape[-1]:
                    # If model outputs single channel, expand to N channels
                    yp = yp.expand(-1, -1, yb.shape[-1])
            
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
                
                if yvp.shape != yvb.shape:
                    if len(yvp.shape) == 2:
                        yvp = yvp.unsqueeze(-1).expand(-1, -1, yvb.shape[-1])
                    elif yvp.shape[-1] != yvb.shape[-1]:
                        yvp = yvp.expand(-1, -1, yvb.shape[-1])
                
                vloss += criterion(yvp, yvb).item()
        vloss /= max(1, len(valid_loader))
        
        scheduler.step(vloss)
        
        if (ep + 1) % 5 == 0 or ep == 0:
            logger.info(f"Epoch {ep+1}/{epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={vloss:.6f}")
        
        model.train()
        
        # Early stopping
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"Early stopping at epoch {ep+1}")
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Evaluation
    logger.info(f"\nEvaluating {model_name}...")
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yp = model(xb)
            
            if yp.shape != yb.shape:
                if len(yp.shape) == 2:
                    yp = yp.unsqueeze(-1).expand(-1, -1, yb.shape[-1])
                elif yp.shape[-1] != yb.shape[-1]:
                    yp = yp.expand(-1, -1, yb.shape[-1])
            
            all_preds.append(yp.cpu().numpy())
            all_trues.append(yb.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)  # [n_samples, pred_len, num_assets]
    all_trues = np.concatenate(all_trues, axis=0)   # [n_samples, pred_len, num_assets]
    
    # Calculate metrics per asset and average
    all_metrics = []
    for asset_idx in range(all_preds.shape[2]):
        asset_preds = all_preds[:, :, asset_idx].reshape(-1, 1)
        asset_trues = all_trues[:, :, asset_idx].reshape(-1, 1)
        
        metrics = calculate_all_metrics(asset_preds, asset_trues)
        metrics['asset_idx'] = asset_idx
        all_metrics.append(metrics)
    
    # Average metrics across all assets
    avg_metrics = {}
    for metric in ['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'Trend_Match']:
        values = [m[metric] for m in all_metrics if metric in m]
        if values:
            avg_metrics[metric] = np.mean(values)
    
    avg_metrics['model'] = model_name
    avg_metrics['source'] = 'multi_asset_benchmark'
    
    logger.info(f"\n{model_name} - Average Metrics:")
    for k, v in avg_metrics.items():
        if k in ['model', 'source']:
            continue
        if k == 'Trend_Match':
            logger.info(f"  {k}: {v:.2f}%")
        else:
            logger.info(f"  {k}: {v:.6f}")
    
    return avg_metrics, all_metrics


def run_multi_asset_benchmark():
    """Run multi-asset benchmark for all models"""
    logger = setup_logging()
    
    # Configuration
    data_dir = 'data'
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
            use_returns=True,
            log_returns=True,
            standardize=True
        )
        
        logger.info(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}, Test batches: {len(test_loader)}")
        
        sample_x, sample_y = next(iter(train_loader))
        logger.info(f"Input shape: {sample_x.shape} (batch, seq_len, num_assets)")
        logger.info(f"Target shape: {sample_y.shape} (batch, pred_len, num_assets)")
        num_assets = sample_x.shape[2]
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        logger.error(traceback.format_exc())
        return
    
    # List of models to test
    models_to_test = []
    all_results = []
    
    # 1. Linear Models
    logger.info("\n" + "="*80)
    logger.info("LINEAR MODELS")
    logger.info("="*80)
    
    for model_name in ['Linear', 'DLinear', 'NLinear', 'RLinear']:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {model_name}")
            logger.info(f"{'='*60}")
            
            if model_name == 'Linear':
                config = LinearConfig()
            elif model_name == 'DLinear':
                config = DLinearConfig()
            elif model_name == 'NLinear':
                config = NLinearConfig()
            elif model_name == 'RLinear':
                config = RLinearConfig()
            
            config.seq_len = seq_len
            config.pred_len = pred_len
            config.enc_in = num_assets  # Multi-asset: N channels
            config.c_out = num_assets
            config.individual = True  # Process each asset individually
            
            if model_name == 'Linear':
                model = Linear(config)
            elif model_name == 'DLinear':
                model = DLinear(config)
            elif model_name == 'NLinear':
                model = NLinear(config)
            elif model_name == 'RLinear':
                model = RLinearModel(config)
            
            metrics, _ = train_and_evaluate_model(
                model, train_loader, valid_loader, test_loader,
                model_name, device, logger, epochs=20
            )
            all_results.append(metrics)
            
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
            logger.error(traceback.format_exc())
            continue
    
    # 2. iTransformer
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing iTransformer")
        logger.info(f"{'='*60}")
        
        config = iTransformerConfig()
        config.seq_len = seq_len
        config.pred_len = pred_len
        config.enc_in = num_assets
        config.c_out = num_assets
        
        model = iTransformer(config)
        metrics, _ = train_and_evaluate_model(
            model, train_loader, valid_loader, test_loader,
            'iTransformer', device, logger, epochs=20
        )
        all_results.append(metrics)
        
    except Exception as e:
        logger.error(f"Error testing iTransformer: {e}")
        logger.error(traceback.format_exc())
    
    # 3. PatchTST
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing PatchTST")
        logger.info(f"{'='*60}")
        
        config = PatchTSTConfig()
        config.seq_len = seq_len
        config.pred_len = pred_len
        config.enc_in = num_assets
        config.c_out = num_assets
        
        model = PatchTST(config)
        metrics, _ = train_and_evaluate_model(
            model, train_loader, valid_loader, test_loader,
            'PatchTST', device, logger, epochs=20
        )
        all_results.append(metrics)
        
    except Exception as e:
        logger.error(f"Error testing PatchTST: {e}")
        logger.error(traceback.format_exc())
    
    # 4. VanillaTransformer
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing VanillaTransformer")
        logger.info(f"{'='*60}")
        
        config = VanillaTransformerConfig()
        config.seq_len = seq_len
        config.pred_len = pred_len
        config.enc_in = num_assets
        config.dec_in = num_assets
        config.c_out = num_assets
        
        model = VanillaTransformer(config)
        metrics, _ = train_and_evaluate_model(
            model, train_loader, valid_loader, test_loader,
            'VanillaTransformer', device, logger, epochs=20
        )
        all_results.append(metrics)
        
    except Exception as e:
        logger.error(f"Error testing VanillaTransformer: {e}")
        logger.error(traceback.format_exc())
    
    # 5. SimpleMoLE
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing SimpleMoLE")
        logger.info(f"{'='*60}")
        
        config = RLGatedMoLEConfig()
        config.seq_len = seq_len
        config.pred_len = pred_len
        config.enc_in = num_assets
        config.c_out = num_assets
        
        model = SimpleMoLE(config)
        metrics, _ = train_and_evaluate_model(
            model, train_loader, valid_loader, test_loader,
            'SimpleMoLE', device, logger, epochs=20
        )
        all_results.append(metrics)
        
    except Exception as e:
        logger.error(f"Error testing SimpleMoLE: {e}")
        logger.error(traceback.format_exc())
    
    # 6. HIEU (already tested, but include for completeness)
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing HIEU")
        logger.info(f"{'='*60}")
        
        cfg = HIEUConfig()
        cfg.seq_len = seq_len
        cfg.pred_len = pred_len
        cfg.num_nodes = num_assets
        cfg.batch_size = batch_size
        cfg.epochs = 30
        cfg.learning_rate = 1e-3
        
        model = HIEUModel(cfg)
        metrics, _ = train_and_evaluate_model(
            model, train_loader, valid_loader, test_loader,
            'HIEU', device, logger, epochs=30
        )
        all_results.append(metrics)
        
    except Exception as e:
        logger.error(f"Error testing HIEU: {e}")
        logger.error(traceback.format_exc())
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        analysis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        results_path = os.path.join(analysis_dir, 'multi_asset_benchmark_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"\n{'='*80}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Total models tested: {len(all_results)}")
        logger.info(f"{'='*80}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("MULTI-ASSET BENCHMARK SUMMARY")
        logger.info("="*80)
        logger.info("\nResults sorted by RMSE:")
        sorted_results = sorted(all_results, key=lambda x: x.get('RMSE', float('inf')))
        for i, result in enumerate(sorted_results, 1):
            logger.info(f"{i}. {result['model']}: RMSE={result.get('RMSE', 'N/A'):.6f}, MAE={result.get('MAE', 'N/A'):.6f}")
    
    logger.info(f"\n{'='*80}")
    logger.info("Multi-Asset Benchmark Completed")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    run_multi_asset_benchmark()

