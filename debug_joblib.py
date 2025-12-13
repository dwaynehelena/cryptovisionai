
import joblib
import os
import sys

try:
    path = "models/ensemble/ensemble_metadata.joblib"
    if os.path.exists(path):
        data = joblib.load(path)
        print("Metadata Content:")
        print(data)
    else:
        print(f"File not found: {path}")
except Exception as e:
    print(f"Error loading file: {e}")
