import subprocess
import sys
import os

PYTHON_EXE = sys.executable

# Absolute paths to your scripts
SCRIPT_1 = r"C:\Harsh\Github\stock_prediction_powerBI.py"
SCRIPT_2 = r"C:\Harsh\Github\prediction.py"

print(" Starting master pipeline...")

# Running stock data extraction script
print(" Running stock_prediction_powerBI.py")
subprocess.run(
    [PYTHON_EXE, SCRIPT_1],
    check=True
)

# Run prediction script
print(" Running prediction.py")
subprocess.run(
    [PYTHON_EXE, SCRIPT_2],
    check=True
)

print(" Master pipeline completed successfully")
