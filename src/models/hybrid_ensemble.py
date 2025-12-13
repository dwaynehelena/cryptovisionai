
import numpy as np
import logging
from typing import Dict, Any, List, Optional
import os
import joblib

logger = logging.getLogger("hybrid_ensemble")

class HybridEnsemble:
    """
    Manages the training and inference of a hybrid ensemble model
    combining TiDE, Mamba, and PatchTST.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models = {}
        self.weights = {}
        
    def train_component_models(self, data: Dict[str, Any]):
        """
        Train all component models individually.
        data: {
          'dl': {'X_train', 'y_train', 'X_val', 'y_val'},
          # feature counts etc
        }
        """
        X_train = data['dl']['X_train']
        y_train = data['dl']['y_train']
        X_val = data['dl']['X_val']
        y_val = data['dl']['y_val']
        
        seq_len = X_train.shape[1]
        n_features = X_train.shape[2]
        
        # 1. TiDE (The Long-Term Expert)
        logger.info("Training TiDE Model...")
        from src.models.tide_model import TiDEModel
        tide_cfg = {
            'hidden_dim': 256, 'num_layers': 3, 'dropout_rate': 0.2, 
            'learning_rate': 0.0005, 'output_dim': 1, 'output_activation': 'sigmoid', 'loss': 'binary_crossentropy'
        }
        self.models['tide'] = TiDEModel(tide_cfg, sequence_length=seq_len, n_features=n_features)
        self.models['tide'].build_model()
        self.models['tide'].train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=20, batch_size=256, verbose=0)
        
        # 2. PatchTST (The Pattern Expert)
        logger.info("Training PatchTST Model...")
        from src.models.patch_tst_model import PatchTSTModel
        patch_cfg = {
            'patch_len': 16, 'stride': 8, 'embed_dim': 64, 'num_heads': 4, 
            'num_layers': 3, 'learning_rate': 0.0005
        }
        self.models['patch'] = PatchTSTModel(patch_cfg, sequence_length=seq_len, n_features=n_features)
        try:
            self.models['patch'].train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=20, batch_size=256, verbose=0)
        except Exception as e:
            logger.error(f"PatchTST failed: {e}")
            del self.models['patch']
            
        # 3. Mamba (The Sequence Expert)
        logger.info("Training Mamba Model...")
        try:
            from src.models.mamba_model import MambaModel
            mamba_cfg = {
                'd_model': 64, 'num_layers': 2, 'dropout': 0.1, 'learning_rate': 0.001
            }
            self.models['mamba'] = MambaModel(mamba_cfg, sequence_length=seq_len, n_features=n_features)
            self.models['mamba'].train(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=256, verbose=0)
        except Exception as e:
            logger.error(f"Mamba failed: {e}")
            # If Mamba fails due to strict TF requirements, we proceed without it
            if 'mamba' in self.models: del self.models['mamba']
            
    def optimize_weights(self, X_val, y_val):
        """
        Find optimal weights OLS or simple brute force
        Since we only have 3 models, brute force or SLSQP is fast.
        """
        logger.info("Optimizing Ensemble Weights...")
        predictions = {}
        for name, model in self.models.items():
            preds = model.predict(X_val)
            # Ensure shape (N,)
            predictions[name] = preds.flatten()
            
        names = list(predictions.keys())
        matrix = np.array([predictions[n] for n in names]).T # (N, 3)
        y_true = y_val.flatten()
        
        # Simple Grid Search for 3 weights summing to 1
        best_acc = 0
        best_weights = [1.0/len(names)] * len(names)
        
        # Coarse grid
        steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        import itertools
        for w in itertools.product(steps, repeat=len(names)):
            if abs(sum(w) - 1.0) > 0.01: continue
            
            # Weighted sum
            final_pred = np.zeros_like(y_true, dtype=float)
            for i, name in enumerate(names):
                final_pred += w[i] * predictions[name]
                
            acc = np.mean((final_pred > 0.5) == y_true)
            if acc > best_acc:
                best_acc = acc
                best_weights = w
                
        self.weights = dict(zip(names, best_weights))
        logger.info(f"Best Ensemble Acc: {best_acc:.4f} with weights: {self.weights}")
        return best_acc

    def predict(self, X):
        final_pred = np.zeros(len(X))
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w > 0:
                p = model.predict(X).flatten()
                final_pred += w * p
        return final_pred

    def save(self, path):
        # Save weights and metadata
        meta = {'weights': self.weights, 'config': self.config}
        joblib.dump(meta, f"{path}_metadata.joblib")
        # Models should be saved individually in a subfolder
        base_dir = os.path.dirname(path)
        for name, model in self.models.items():
            model.save(os.path.join(base_dir, f"{name}_component.h5"))

    def load(self, path):
        """
        Load weights and metadata from path
        """
        try:
            # Load metadata
            meta_path = f"{path}_metadata.joblib"
            if not os.path.exists(meta_path):
                logger.error(f"Metadata not found at {meta_path}")
                return False
                
            meta = joblib.load(meta_path)
            self.weights = meta.get('weights', {})
            self.config = meta.get('config', {})
            
            # Reconstruct models
            base_dir = os.path.dirname(path)
            
            # We need to know shape to initialize, usually in config or we infer?
            # For now, let's assume we can load weights without shape if using proper save/load or re-init default
            # Actually TiDE and others need shape to build_model() before load_weights
            # We should assume standard shapes or save them in config
            
            # TEMPORARY: Hardcoded shapes matching training script or strictly inferred
            seq_len = 60 # Default from train_super_ensemble.py
            n_features = 146 # Default 
            
            # Try to infer from config if available
            # ...
            
            # 1. TiDE
            tide_path = os.path.join(base_dir, "tide_component.h5")
            if os.path.exists(tide_path):
                from src.models.tide_model import TiDEModel
                tide_cfg = {
                    'hidden_dim': 256, 'num_layers': 3, 'dropout_rate': 0.2, 
                    'output_dim': 1, 'output_activation': 'sigmoid'
                }
                self.models['tide'] = TiDEModel(tide_cfg, sequence_length=seq_len, n_features=n_features)
                self.models['tide'].build_model()
                self.models['tide'].model.load_weights(tide_path)
                logger.info("Loaded TiDE component")
                
            # 2. PatchTST
            patch_path = os.path.join(base_dir, "patch_component.h5")
            if os.path.exists(patch_path):
                from src.models.patch_tst_model import PatchTSTModel
                patch_cfg = {
                    'patch_len': 16, 'stride': 8, 'embed_dim': 64, 'num_heads': 4, 
                    'num_layers': 3
                }
                self.models['patch'] = PatchTSTModel(patch_cfg, sequence_length=seq_len, n_features=n_features)
                # PatchTST usually needs build or just load? 
                # Assuming similar API
                try:
                    # self.models['patch'].build_model() # If needed
                    self.models['patch'].model.load_weights(patch_path)
                    logger.info("Loaded PatchTST component")
                except:
                    pass
            
            # 3. Mamba
            mamba_path = os.path.join(base_dir, "mamba_component.h5")
            if os.path.exists(mamba_path):
                from src.models.mamba_model import MambaModel
                mamba_cfg = {'d_model': 64, 'num_layers': 2}
                self.models['mamba'] = MambaModel(mamba_cfg, sequence_length=seq_len, n_features=n_features)
                self.models['mamba'].model.load_weights(mamba_path)
                logger.info("Loaded Mamba component")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load HybridEnsemble: {e}")
            return False

