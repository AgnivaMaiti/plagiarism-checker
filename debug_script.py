#!/usr/bin/env python3
import subprocess
import sys
import os

print("=== DEBUG SCRIPT STARTED ===")

# Try to run a simple command to see if we get output
try:
    result = subprocess.run([sys.executable, '-c', 'print("Hello from Python")'], 
                          capture_output=True, text=True, timeout=10)
    print(f"Command stdout: {result.stdout}")
    print(f"Command stderr: {result.stderr}")
    print(f"Return code: {result.returncode}")
except Exception as e:
    print(f"Error running command: {e}")

# Check if files exist
print("\n=== FILE CHECK ===")
files_to_check = ['plagiarism_dataset.csv', 'app.py', 'main.py', 'requirements.txt']
for file in files_to_check:
    exists = os.path.exists(file)
    print(f"{file}: {'✅ EXISTS' if exists else '❌ MISSING'}")

# Check if we can import required modules
print("\n=== MODULE IMPORT CHECK ===")
modules_to_check = ['flask', 'pandas', 'sentence_transformers', 'faiss', 'xgboost']
for module in modules_to_check:
    try:
        __import__(module)
        print(f"{module}: ✅ IMPORT SUCCESS")
    except ImportError as e:
        print(f"{module}: ❌ IMPORT FAILED - {e}")

print("=== DEBUG SCRIPT COMPLETED ===")
