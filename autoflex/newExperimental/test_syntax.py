#!/usr/bin/env python3
"""
Syntax validation for experimental vision transformer implementations.
This script checks if all files have valid Python syntax without importing heavy dependencies.
"""

import ast
import os
import sys


def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Parse the AST to check syntax
        ast.parse(content)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Check syntax of all experimental files."""
    experimental_files = [
        'fourier_attention.py',
        'linear_attention.py',
        'vision_mamba.py',
        'kan_transformer.py',
        'hybrid_architectures.py',
        'experimental_vit.py'
    ]
    
    print("Checking syntax of experimental vision transformer implementations...")
    print("=" * 70)
    
    all_valid = True
    
    for file_name in experimental_files:
        if os.path.exists(file_name):
            is_valid, message = check_syntax(file_name)
            status = "✓ PASS" if is_valid else "✗ FAIL"
            print(f"{file_name:25} | {status:8} | {message}")
            
            if not is_valid:
                all_valid = False
        else:
            print(f"{file_name:25} | ✗ FAIL   | File not found")
            all_valid = False
    
    print("=" * 70)
    
    if all_valid:
        print("✓ All files have valid syntax!")
        return 0
    else:
        print("✗ Some files have syntax errors!")
        return 1


if __name__ == "__main__":
    sys.exit(main())