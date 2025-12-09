import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

print("Attempting to import src.api.main...")
try:
    from src.api.main import app
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
