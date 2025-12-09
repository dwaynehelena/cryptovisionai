#!/usr/bin/env python3
"""
Main Entry Point for CryptoVisionAI
"""

import sys
import os

# Ensure the src directory is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the application
from src.app import CryptoVisionAI, parse_args

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Create the application
        app = CryptoVisionAI(args.config)
        
        # Override settings from command line if provided
        if args.mode:
            app.config["general"]["mode"] = args.mode
        if args.debug:
            app.config["general"]["debug"] = True
        
        # Initialize and start the application
        app.initialize_components()
        app.start()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")