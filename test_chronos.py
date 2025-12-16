import sys
import os
import logging
import json

# Add project root to path
sys.path.append(os.getcwd())

from src.models.chronos_model import ChronosModel

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    print("Testing Chronos T5 Tiny Loading...")
    
    # Initialize
    chronos = ChronosModel(model_name="amazon/chronos-t5-tiny")
    
    # Load
    success = chronos.load_from_hub()
    
    if success:
        print("\nSUCCESS: Model loaded.")
        
        # Print Config
        config = chronos.get_config_dict()
        print("\n--- Model Configuration ---")
        # Filter some keys for readability
        keys_to_show = ['architectures', 'd_model', 'num_layers', 'num_heads', 'vocab_size', 'model_type']
        filtered_config = {k: config.get(k) for k in keys_to_show}
        print(json.dumps(filtered_config, indent=2))
        
        print("\nFull config available in 'config' attribute.")
    else:
        print("\nFAILURE: Model could not be loaded.")
        sys.exit(1)

if __name__ == "__main__":
    main()
