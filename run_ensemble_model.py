import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.ensemble_model import EnsembleModel
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing Ensemble Model...")
    model = EnsembleModel()
    
    # Generate dummy data
    print("Generating dummy training data...")
    n_samples = 1000
    n_features = 20
    X_train = np.random.rand(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    data_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_train[:100],
        'y_val': y_train[:100]
    }
    
    model_configs = {
        # 'random_forest': {'n_estimators': 10},
        # 'xgboost': {'n_estimators': 10},
        'lightgbm': {'n_estimators': 10} # Using only lightgbm for speed and likely availability
    }
    
    save_dir = "models/ensemble"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Training models...")
    # Need to verify if xgboost/lightgbm are installed in the env. 
    # The previous pip install output showed they might be (implied by file imports existing).
    # I'll try with just Random Forest first if others fail, but let's try all or subset.
    # Actually, for robustness, I'll stick to RandomForest if I'm not sure, but the code imports xgboost/lightgbm.
    # Let's enable Random Forest as it's standard sklearn.
    
    model_configs = {
        'random_forest': {'n_estimators': 10},
    }
    
    model.train_base_models(data_dict, model_configs, save_dir=save_dir)
    
    # The TradingSystem expects a 'voting' classifier by default or just individual models.
    # The `EnsembleModel.save` method saves metadata. 
    # train_base_models calls save logic if save_dir is passed.
    
    # Also explicitly call save just in case
    model.save(save_dir)
    print(f"Model saved to {save_dir}")

if __name__ == "__main__":
    main()
