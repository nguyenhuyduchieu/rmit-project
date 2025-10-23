"""
Test script for the time series forecasting framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessing import CryptoDataPreprocessor
from src.data.dataset import create_data_loaders
from src.utils.metrics import calculate_all_metrics
from src.configs.model_configs import LinearConfig
from src.models.linear_models import Linear
from src.utils.training import ModelTrainer

def test_data_preprocessing():
    """Test data preprocessing pipeline"""
    print("Testing data preprocessing...")
    
    data_dir = '/Users/hieuduc/Downloads/rmit/data'
    preprocessor = CryptoDataPreprocessor(data_dir)
    
    # Test with a single file
    test_file = 'BTCUSDT.csv'
    file_path = os.path.join(data_dir, test_file)
    
    try:
        train_data, valid_data, test_data = preprocessor.process_single_file(
            file_path, max_samples=10000
        )
        
        print(f"✓ Data preprocessing successful")
        print(f"  Train data shape: {train_data.shape}")
        print(f"  Valid data shape: {valid_data.shape}")
        print(f"  Test data shape: {test_data.shape}")
        
        return train_data, valid_data, test_data
        
    except Exception as e:
        print(f"✗ Data preprocessing failed: {e}")
        return None, None, None

def test_model_training():
    """Test model training pipeline"""
    print("\nTesting model training...")
    
    # Get preprocessed data
    train_data, valid_data, test_data = test_data_preprocessing()
    
    if train_data is None:
        print("✗ Cannot test model training - data preprocessing failed")
        return
    
    try:
        # Create data loaders
        train_loader, valid_loader, test_loader = create_data_loaders(
            train_data, valid_data, test_data,
            seq_len=96, pred_len=24, batch_size=32
        )
        
        # Initialize model
        config = LinearConfig()
        model = Linear(config)
        
        # Train model
        trainer = ModelTrainer(model)
        training_results = trainer.train(
            train_loader, valid_loader, 
            epochs=3, learning_rate=0.001
        )
        
        print(f"✓ Model training successful")
        print(f"  Final train loss: {training_results['train_losses'][-1]:.6f}")
        print(f"  Final val loss: {training_results['val_losses'][-1]:.6f}")
        
        return trainer, test_loader
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return None, None

def test_model_evaluation():
    """Test model evaluation"""
    print("\nTesting model evaluation...")
    
    trainer, test_loader = test_model_training()
    
    if trainer is None:
        print("✗ Cannot test model evaluation - training failed")
        return
    
    try:
        # Evaluate model
        preds, trues = trainer.evaluate(test_loader)
        
        # Calculate metrics
        metrics = calculate_all_metrics(preds, trues)
        
        print(f"✓ Model evaluation successful")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        
    except Exception as e:
        print(f"✗ Model evaluation failed: {e}")

def main():
    """Run all tests"""
    print("="*60)
    print("TESTING TIME SERIES FORECASTING FRAMEWORK")
    print("="*60)
    
    # Test data preprocessing
    test_data_preprocessing()
    
    # Test model training
    test_model_training()
    
    # Test model evaluation
    test_model_evaluation()
    
    print("\n" + "="*60)
    print("FRAMEWORK TESTING COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
