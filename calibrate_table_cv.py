#!/usr/bin/env python3
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our new calibration module
from camera_test.table_calibration_cv import main as calibrate

if __name__ == "__main__":
    print("Starting table calibration using OpenCV triangulation method...")
    calibrate() 