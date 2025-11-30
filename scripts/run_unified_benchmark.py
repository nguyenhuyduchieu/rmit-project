"""
Unified benchmark script using prepared data
All models use the same prepared data for fair comparison
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


def setup_logging(log_file='unified_benchmark.log'):
    """Setup logging to both file and console"""
    # Create logs directory if not exists
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, log_file)
    
    # Configure logging
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
    logger.info(f"Starting Unified Benchmark")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"="*80)
    
    return logger

from data_prepare.load_prepared_data import load_prepared_data, create_data_loaders_from_prepared
from src.configs.model_configs import (
    LinearConfig, DLinearConfig, NLinearConfig, RLinearConfig,
    PatchTSTConfig, iTransformerConfig, VanillaTransformerConfig, 
    RLGatedMoLEConfig, ProphetConfig, AutoformerConfig
)
from baseline_models.linear_models import Linear, DLinear, NLinear, RLinearOLS
from baseline_models.transformer_models import PatchTST, iTransformer, VanillaTransformer
from baseline_models.prophet_model import ProphetModel
from baseline_models.autoformer_model import Autoformer
from scripts.test_simple_mole import SimpleMoLE
from models.HIEU.model import HIEUModel
from models.HIEU.configs import HIEUConfig
from src.utils.training import ModelTrainer
from src.utils.metrics import calculate_all_metrics


def run_single_model_benchmark(
    model_class, config, train_loader, valid_loader, test_loader,
    model_name: str, dataset_name: str, metadata: dict, logger=None
):
    """Run benchmark for a single model using prepared data"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name} on {dataset_name}")
    logger.info(f"{'='*60}")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seq_len = metadata.get('seq_len', 96)
        pred_len = metadata.get('pred_len', 96)
        
        # Update config with actual seq_len and pred_len
        if config is not None:
            if hasattr(config, 'seq_len'):
                config.seq_len = seq_len
            if hasattr(config, 'pred_len'):
                config.pred_len = pred_len
            if hasattr(config, 'enc_in'):
                config.enc_in = metadata.get('num_features', 1)
            if hasattr(config, 'label_len') and config.label_len is None:
                config.label_len = seq_len // 2
        
        if model_name in ('Linear', 'DLinear', 'NLinear'):
            logger.info(f"Initializing {model_name} model...")
            model = model_class(config)
            logger.info(f"Training {model_name}...")
            trainer = ModelTrainer(model)
            trainer.train(train_loader, valid_loader, 
                         epochs=getattr(config, 'epochs', 5), 
                         learning_rate=getattr(config, 'learning_rate', 1e-3))
            logger.info(f"Evaluating {model_name}...")
            preds, trues = trainer.evaluate(test_loader)
            all_preds, all_trues = preds, trues
            
        elif model_name == 'RLinear':
            # RLinearOLS requires context_length, horizon and dataset_train
            cl = seq_len
            hz = pred_len
            alpha = getattr(config, 'alpha', 1e-6)
            
            # Extract training series (first feature = Close)
            train_data = metadata.get('train_data_raw', None)
            if train_data is None:
                # Need to get from loader - simplified approach
                series_train = []
                for batch_x, _ in train_loader:
                    series_train.append(batch_x[:, :, 0].cpu().numpy())
                series_train = np.concatenate(series_train, axis=0).flatten()
            else:
                series_train = train_data[:, 0]
            
            dataset_train = series_train.reshape(1, -1).astype(np.float32)
            
            try:
                if dataset_train.shape[1] < cl + hz:
                    raise ValueError('Not enough training length for RLinearOLS')
                model = RLinearOLS(context_length=cl, horizon=hz, 
                                  dataset_train=dataset_train, alpha=alpha)
                trainer = ModelTrainer(model)
                trainer.train(train_loader, valid_loader, 
                             epochs=getattr(config, 'epochs', 5), 
                             learning_rate=getattr(config, 'learning_rate', 1e-3))
                preds, trues = trainer.evaluate(test_loader)
                all_preds, all_trues = preds, trues
            except Exception as e:
                logger.warning(f"RLinearOLS failed, using fallback: {e}")
                from baseline_models.rlinear_model import RLinearModel as TorchRLinear
                class Cfg: pass
                cfg = Cfg()
                cfg.seq_len = cl
                cfg.pred_len = hz
                cfg.enc_in = 1
                cfg.individual = True
                model = TorchRLinear(cfg)
                trainer = ModelTrainer(model)
                trainer.train(train_loader, valid_loader, epochs=5, learning_rate=1e-3)
                preds, trues = trainer.evaluate(test_loader)
                all_preds, all_trues = preds, trues
                
        elif model_name == 'PatchTST':
            logger.info(f"Initializing {model_name}...")
            model = model_class(config)
            trainer = ModelTrainer(model)
            logger.info(f"Training {model_name}...")
            trainer.train(train_loader, valid_loader, 
                         epochs=config.epochs, 
                         learning_rate=config.learning_rate)
            logger.info(f"Evaluating {model_name}...")
            preds, trues = trainer.evaluate(test_loader)
            all_preds, all_trues = preds, trues
            
        elif model_name == 'Autoformer':
            # Try to use Autoformer, skip if not available
            try:
                logger.info(f"Initializing {model_name}...")
                model = model_class(config)
                trainer = ModelTrainer(model)
                logger.info(f"Training {model_name}...")
                trainer.train(train_loader, valid_loader, 
                             epochs=config.epochs, 
                             learning_rate=config.learning_rate)
                logger.info(f"Evaluating {model_name}...")
                preds, trues = trainer.evaluate(test_loader)
                all_preds, all_trues = preds, trues
            except (ImportError, AttributeError) as e:
                logger.warning(f"Autoformer not available: {e}. Skipping...")
                raise  # Re-raise to be caught by outer exception handler
            
        elif model_name == 'iTransformer':
            # Custom train/eval for iTransformer
            model = model_class(config)
            model = model.to(device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            
            # Train
            logger.info(f"Training {model_name}...")
            model.train()
            for _ in range(config.epochs):
                for batch_x, batch_y in train_loader:
                    # Use first feature (Close) as 1-channel input
                    batch_x = batch_x[:, :, :1].to(device)
                    batch_y = batch_y[:, :, :1].to(device)
                    B, L, C = batch_x.shape
                    _, H, _ = batch_y.shape
                    x_mark_enc = torch.zeros(B, L, 0, device=device)
                    x_mark_dec = torch.zeros(B, H, 0, device=device)
                    x_dec = torch.zeros(B, H, C, device=device)
                    outputs = model(batch_x, x_mark_enc, x_dec, x_mark_dec)
                    loss = criterion(outputs, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Eval
            logger.info(f"Evaluating {model_name}...")
            model.eval()
            preds_list, trues_list = [], []
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x[:, :, :1].to(device)
                    batch_y = batch_y[:, :, :1].to(device)
                    B, L, C = batch_x.shape
                    _, H, _ = batch_y.shape
                    x_mark_enc = torch.zeros(B, L, 0, device=device)
                    x_mark_dec = torch.zeros(B, H, 0, device=device)
                    x_dec = torch.zeros(B, H, C, device=device)
                    outputs = model(batch_x, x_mark_enc, x_dec, x_mark_dec)
                    preds_list.append(outputs.cpu().numpy())
                    trues_list.append(batch_y.cpu().numpy())
            all_preds = np.concatenate(preds_list, axis=0)
            all_trues = np.concatenate(trues_list, axis=0)
            
        elif model_name == 'VanillaTransformer':
            # VanillaTransformer needs enc_in=1 config
            logger.info(f"Initializing {model_name} with enc_in=1...")
            # Update config to use only 1 channel
            config.enc_in = 1
            config.dec_in = 1
            config.c_out = 1
            model = model_class(config)
            model = model.to(device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            
            # Train
            logger.info(f"Training {model_name}...")
            model.train()
            for _ in range(config.epochs):
                for batch_x, batch_y in train_loader:
                    # Use first feature (Close) as 1-channel input
                    batch_x = batch_x[:, :, :1].to(device)
                    batch_y = batch_y[:, :, :1].to(device)
                    B, L, C = batch_x.shape
                    _, H, _ = batch_y.shape
                    x_mark_enc = torch.zeros(B, L, 0, device=device)
                    x_mark_dec = torch.zeros(B, H, 0, device=device)
                    x_dec = torch.zeros(B, H, C, device=device)
                    outputs = model(batch_x, x_mark_enc, x_dec, x_mark_dec)
                    loss = criterion(outputs, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            
            # Eval
            logger.info(f"Evaluating {model_name}...")
            model.eval()
            preds_list, trues_list = [], []
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x[:, :, :1].to(device)
                    batch_y = batch_y[:, :, :1].to(device)
                    B, L, C = batch_x.shape
                    _, H, _ = batch_y.shape
                    x_mark_enc = torch.zeros(B, L, 0, device=device)
                    x_mark_dec = torch.zeros(B, H, 0, device=device)
                    x_dec = torch.zeros(B, H, C, device=device)
                    outputs = model(batch_x, x_mark_enc, x_dec, x_mark_dec)
                    preds_list.append(outputs.cpu().numpy())
                    trues_list.append(batch_y.cpu().numpy())
            all_preds = np.concatenate(preds_list, axis=0)
            all_trues = np.concatenate(trues_list, axis=0)
            
        elif model_name == 'SimpleMoLE':
            logger.info(f"Initializing {model_name}...")
            model = model_class(config)
            logger.info(f"Training {model_name}...")
            model.train_model(train_loader, valid_loader, expert_epochs=3, router_epochs=3)
            model.train(False)
            logger.info(f"Evaluating {model_name}...")
            test_predictions = []
            test_targets = []
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    pred = model(batch_x)  # Shape: [batch, pred_len]
                    # Reshape to match target shape [batch, pred_len, 1]
                    if len(pred.shape) == 2:
                        pred = pred.unsqueeze(-1)  # Add channel dimension
                    test_predictions.append(pred.cpu().numpy())
                    test_targets.append(batch_y.cpu().numpy())
            all_preds = np.concatenate(test_predictions, axis=0)
            all_trues = np.concatenate(test_targets, axis=0)
            
        elif model_name == 'Prophet':
            # Prophet needs special handling
            logger.info(f"Initializing {model_name}...")
            model = ProphetModel(config)
            
            # Extract full training and validation data for Prophet
            logger.info(f"Preparing data for {model_name}...")
            train_full = []
            valid_full = []
            for batch_x, _ in train_loader:
                train_full.append(batch_x.numpy())
            for batch_x, _ in valid_loader:
                valid_full.append(batch_x.numpy())
            train_full = np.concatenate(train_full, axis=0)
            valid_full = np.concatenate(valid_full, axis=0)
            
            # Extract Close price (first feature) from last timestep of each window
            train_series = train_full[:, -1, 0]  # [n_samples]
            valid_series = valid_full[:, -1, 0]   # [n_samples]
            
            # Combine train and valid
            combined_series = np.concatenate([train_series, valid_series])
            
            logger.info(f"Training {model_name} on {len(combined_series)} samples...")
            model.train_on_full_series(combined_series.reshape(-1, 1), None)
            
            logger.info(f"Evaluating {model_name}...")
            all_preds, all_trues = model.predict_sliding_windows(test_loader)
            
        elif model_name == 'HIEU':
            # HIEU model - improved training
            logger.info(f"Initializing {model_name} with improved config...")
            cfg = HIEUConfig()
            cfg.seq_len = seq_len
            cfg.pred_len = pred_len
            cfg.num_nodes = 1  # Single asset
            cfg.batch_size = 32
            cfg.epochs = 30  # Increased epochs
            cfg.learning_rate = 5e-4  # Lower learning rate for stability
            cfg.weight_decay = 1e-5
            
            # Additional improvements
            if hasattr(cfg, 'use_revin'):
                cfg.use_revin = True  # Use RevIN normalization if available
            
            model = HIEUModel(cfg)
            model = model.to(device)
            
            # Use AdamW with better settings
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=cfg.learning_rate, 
                weight_decay=cfg.weight_decay,
                betas=(0.9, 0.999)
            )
            
            # Use ReduceLROnPlateau scheduler for better convergence
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
            
            criterion = torch.nn.MSELoss()
            best_val = float('inf')
            best_state = None
            patience = 8  # Increased patience
            bad = 0
            
            logger.info(f"Training {model_name} for {cfg.epochs} epochs...")
            # Train
            model.train()
            for ep in range(cfg.epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for xb, yb in train_loader:
                    # HIEU expects [B, L, N] - use first feature (Close)
                    xb = xb[:, :, :1].to(device)  # [B, L, 1]
                    yb = yb.to(device)  # [B, H, 1]
                    
                    optimizer.zero_grad()
                    yp = model(xb)  # [B, H, 1]
                    loss = criterion(yp, yb)
                    loss.backward()
                    
                    # Gradient clipping for stability
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
                        xvb = xvb[:, :, :1].to(device)
                        yvb = yvb.to(device)
                        yvp = model(xvb)
                        vloss += criterion(yvp, yvb).item()
                vloss /= max(1, len(valid_loader))
                
                # Update learning rate
                scheduler.step(vloss)
                
                # Log progress
                if (ep + 1) % 5 == 0 or ep == 0:
                    logger.info(f"  Epoch {ep+1}/{cfg.epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={vloss:.6f}, LR={optimizer.param_groups[0]['lr']:.6f}")
                
                model.train()
                
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
            
            if best_state is not None:
                logger.info(f"Loading best model with val loss: {best_val:.6f}")
                model.load_state_dict(best_state)
            
            # Eval
            logger.info(f"Evaluating {model_name}...")
            model.eval()
            preds = []
            trues = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb[:, :, :1].to(device)
                    yp = model(xb)
                    preds.append(yp.cpu().numpy())
                    trues.append(yb.cpu().numpy())
            all_preds = np.concatenate(preds, axis=0)
            all_trues = np.concatenate(trues, axis=0)
            
        else:
            # Generic model training
            model = model_class(config)
            trainer = ModelTrainer(model)
            trainer.train(train_loader, valid_loader, 
                         epochs=getattr(config, 'epochs', 5), 
                         learning_rate=getattr(config, 'learning_rate', 1e-3))
            preds, trues = trainer.evaluate(test_loader)
            all_preds, all_trues = preds, trues
        
        # Calculate metrics
        logger.info(f"Calculating metrics for {model_name}...")
        metrics = calculate_all_metrics(all_preds, all_trues)
        
        logger.info(f"\nResults for {model_name}:")
        for k, v in metrics.items():
            if k == 'Trend_Match':
                logger.info(f"  {k}: {v:.2f}%")
            else:
                logger.info(f"  {k}: {v:.6f}")
        
        logger.info(f"✓ {model_name} completed successfully")
        return {'model': model_name, 'dataset': dataset_name, 'metrics': metrics, 'success': True}
        
    except Exception as e:
        logger.error(f"✗ Error testing {model_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {'model': model_name, 'dataset': dataset_name, 'metrics': {}, 'success': False, 'error': str(e)}


def main():
    """Main benchmark function"""
    # Setup logging
    logger = setup_logging('unified_benchmark.log')
    
    logger.info("="*80)
    logger.info("UNIFIED BENCHMARK WITH PREPARED DATA")
    logger.info("All models use the same prepared data for fair comparison")
    logger.info("="*80)
    
    # Configuration
    data_prepare_dir = 'data_prepare'
    datasets = ['BTCUSDT']  # Can be extended
    seq_len = 96
    pred_len = 96
    batch_size = 32
    
    logger.info(f"Configuration:")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Sequence length: {seq_len}")
    logger.info(f"  Prediction length: {pred_len}")
    logger.info(f"  Batch size: {batch_size}")
    
    all_results = []
    
    for dataset_name in datasets:
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING DATASET: {dataset_name}")
        logger.info(f"{'='*80}")
        
        # Load prepared data
        try:
            logger.info(f"Loading prepared data for {dataset_name}...")
            prepared_data = load_prepared_data(dataset_name, data_prepare_dir)
            train_data = prepared_data['train_data']
            valid_data = prepared_data['valid_data']
            test_data = prepared_data['test_data']
            metadata = prepared_data['metadata']
            metadata['seq_len'] = seq_len
            metadata['pred_len'] = pred_len
            
            logger.info(f"Loaded prepared data:")
            logger.info(f"  Train: {train_data.shape}")
            logger.info(f"  Valid: {valid_data.shape}")
            logger.info(f"  Test: {test_data.shape}")
            logger.info(f"  Features: {metadata.get('num_features', 'unknown')}")
            
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            logger.error(f"Please run: python data_prepare/prepare_data.py --datasets {dataset_name}")
            continue
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, valid_loader, test_loader = create_data_loaders_from_prepared(
            train_data, valid_data, test_data,
            seq_len, pred_len, batch_size,
            target_col_idx=0  # Close price is first feature
        )
        logger.info(f"Created data loaders: train={len(train_loader)}, valid={len(valid_loader)}, test={len(test_loader)}")
        
        # Define models to test
        models_to_test = [
            ('Linear', Linear, LinearConfig()),
            ('DLinear', DLinear, DLinearConfig()),
            ('NLinear', NLinear, NLinearConfig()),
            ('RLinear', RLinearOLS, RLinearConfig()),
            ('PatchTST', PatchTST, PatchTSTConfig()),
            ('iTransformer', iTransformer, iTransformerConfig()),
            ('VanillaTransformer', VanillaTransformer, VanillaTransformerConfig()),
            ('SimpleMoLE', SimpleMoLE, RLGatedMoLEConfig()),
            ('Prophet', ProphetModel, ProphetConfig()),
            ('Autoformer', Autoformer, AutoformerConfig()),
            ('HIEU', HIEUModel, None)
        ]
        
        logger.info(f"Testing {len(models_to_test)} models: {[m[0] for m in models_to_test]}")
        
        # Test each model
        for idx, (model_name, model_class, config) in enumerate(models_to_test, 1):
            logger.info(f"\n[{idx}/{len(models_to_test)}] Starting {model_name}...")
            result = run_single_model_benchmark(
                model_class, config, train_loader, valid_loader, test_loader,
                model_name, dataset_name, metadata, logger
            )
            if result['success']:
                all_results.append(result)
                logger.info(f"✓ {model_name} completed and added to results")
            else:
                logger.warning(f"✗ {model_name} failed: {result.get('error', 'Unknown error')}")
    
    # Save results
    logger.info(f"\n{'='*80}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'='*80}")
    
    if all_results:
        logger.info(f"Total successful experiments: {len(all_results)}")
        results_df = pd.DataFrame(all_results)
        metrics_df = pd.json_normalize(results_df['metrics'])
        results_df = pd.concat([results_df.drop('metrics', axis=1), metrics_df], axis=1)
        
        analysis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        results_path = os.path.join(analysis_dir, 'unified_benchmark_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved detailed results to: {results_path}")
        
        # Calculate summary
        summary_data = []
        for model in results_df['model'].unique():
            model_results = results_df[results_df['model'] == model]
            row = {'model': model}
            for metric in ['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'Trend_Match']:
                if metric in model_results.columns:
                    row[metric] = model_results[metric].mean()
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(analysis_dir, 'unified_benchmark_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary to: {summary_path}")
        
        logger.info(f"\n{'='*80}\nBENCHMARK SUMMARY\n{'='*80}")
        logger.info("\n" + summary_df.to_string(index=False))
        logger.info(f"\nResults saved to:")
        logger.info(f"  - {results_path}")
        logger.info(f"  - {summary_path}")
        logger.info(f"\n{'='*80}")
        logger.info("BENCHMARK COMPLETED SUCCESSFULLY!")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}")
    else:
        logger.error("No successful experiments completed.")
        logger.error("Please check the logs above for error details.")


if __name__ == "__main__":
    main()

