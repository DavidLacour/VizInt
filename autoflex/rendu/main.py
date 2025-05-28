#!/usr/bin/env python3
"""
Main entry point for the unified training and evaluation system
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import and run the main function from src/main.py
import main as src_main

if __name__ == "__main__":
    src_main.main()