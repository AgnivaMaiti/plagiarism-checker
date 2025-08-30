#!/usr/bin/env python3
import os
import sys

print("Testing Python execution...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

# Check if required files exist
files_to_check = ['plagiarism_dataset.csv', 'app.py', 'requirements.txt']
for file in files_to_check:
    exists = os.path.exists(file)
    print(f"{file}: {'✅ EXISTS' if exists else '❌ MISSING'}")
