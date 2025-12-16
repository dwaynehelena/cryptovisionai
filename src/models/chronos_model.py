"""
Chronos: Foundation Model for Time Series Forecasting

Wraps the Hugging Face implementation of Chronos (T5-based).
"""

import torch
import logging
from typing import Optional, List, Dict, Any
from transformers import AutoModelForSeq2SeqLM, AutoConfig

logger = logging.getLogger("models.chronos")

class ChronosModel:
    """
    Chronos Foundation Model Wrapper
    
    Loads amazon/chronos-t5 models using Hugging Face Transformers.
    """
    
    def __init__(self, model_name: str = "amazon/chronos-t5-tiny", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.config = None
        
    def load_from_hub(self):
        """
        Load model and config from Hugging Face Hub
        """
        try:
            logger.info(f"Loading Chronos model: {self.model_name}...")
            
            # Load Config
            self.config = AutoConfig.from_pretrained(self.model_name)
            logger.info(f"Config loaded: {self.config}")
            
            # Load Model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32  # Use float32 for CPU/compatibility
            ).to(self.device)
            
            logger.info("Model loaded successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Chronos model: {e}")
            return False

    def get_config_dict(self) -> Dict[str, Any]:
        """Return model configuration as dictionary"""
        if self.config:
            return self.config.to_dict()
        return {}

    def predict(self, context_tensor):
        """
        Raw prediction (generate). 
        
        Args:
            context_tensor: Torch tensor of input tokens [batch, seq_len]
            
        Returns:
            Output tokens
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_from_hub() first.")
            
        with torch.no_grad():
            output = self.model.generate(
                context_tensor.to(self.device),
                max_length=self.config.n_positions if hasattr(self.config, 'n_positions') else 512,
                num_return_sequences=1
            )
        return output

    @staticmethod
    def simple_tokenize(series: List[float], n_tokens=4096) -> torch.Tensor:
        """
        Very basic linear quantization for demonstration.
        Real Chronos uses specific binning logic.
        """
        # Placeholder: Map 0-1 to 0-n_tokens
        # This is NOT the actual Chronos tokenizer, just a placeholder to allow 'running' the model
        # if the user manually feeds data.
        arr = torch.tensor(series)
        # Normalize
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val > 0:
            arr = (arr - min_val) / (max_val - min_val)
        
        tokens = (arr * (n_tokens - 1)).long()
        return tokens.unsqueeze(0) # Batch dim
