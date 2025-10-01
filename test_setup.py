import os
import numpy as np
from pathlib import Path

def test_setup():
    print("=== Testing Setup ===")
    
    # Check current directory
    current_dir = Path(".")
    print(f"Current directory: {current_dir.absolute()}")
    
    # List files
    print("\nFiles in current directory:")
    for file in current_dir.iterdir():
        print(f"  {file.name}")
    
    # Check for bag files
    bag_files = list(current_dir.glob("*.db3"))
    metadata_files = list(current_dir.glob("*.yaml")) + list(current_dir.glob("*.yml"))
    
    print(f"\nFound {len(bag_files)} .db3 files: {[f.name for f in bag_files]}")
    print(f"Found {len(metadata_files)} metadata files: {[f.name for f in metadata_files]}")
    
    # Test imports
    print("\n=== Testing Imports ===")
    try:
        import numpy as np
        import pandas as pd
        import yaml
        from rosbags.rosbag2 import Reader
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")

if __name__ == "__main__":
    test_setup()