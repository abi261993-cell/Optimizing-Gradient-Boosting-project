#!/usr/bin/env python3
"""
run_example.py - quick demo runner for pure_numpy_gbm.py
"""
import subprocess
import sys
import os

SCRIPT = "pure_numpy_gbm.py"

def main():
    if not os.path.exists(SCRIPT):
        print(f"Error: {SCRIPT} not found. Run from repo root.")
        return
    cmd = [sys.executable, SCRIPT, "--quick"]
    print("Running quick demo (this will be brief)...")
    subprocess.run(cmd, check=True)

if _name_ == "_main_":
    main()
