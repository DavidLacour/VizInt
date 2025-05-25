"""
Script to check and summarize which model combinations are evaluated in each main script
"""

import os
import re

def check_file_for_combinations(filepath):
    """Check a file for model combination evaluations"""
    
    combinations_needed = [
        "main model",
        "main model robust", 
        "blendedTTT",
        "blendedTTT3fc",
        "healer + main",
        "healer + robust",
        "ttt + main",
        "ttt + robust", 
        "ttt3fc + main",
        "ttt3fc + robust"
    ]
    
    found_combinations = {}
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Check for each combination
        for combo in combinations_needed:
            # Create search patterns
            patterns = []
            
            if combo == "main model":
                patterns = [r"main.*model", r"evaluate.*main", r"Main Model"]
            elif combo == "main model robust":
                patterns = [r"robust.*model", r"model.*robust", r"Robust.*Model"]
            elif combo == "blendedTTT":
                patterns = [r"blended.*ttt(?!3fc)", r"BlendedTTT(?!3fc)"]
            elif combo == "blendedTTT3fc":
                patterns = [r"blended.*ttt.*3fc", r"BlendedTTT3fc"]
            elif combo == "healer + main":
                patterns = [r"healer.*main", r"main.*healer", r"Healer.*Main"]
            elif combo == "healer + robust":
                patterns = [r"healer.*robust", r"robust.*healer", r"Healer.*Robust"]
            elif combo == "ttt + main":
                patterns = [r"ttt.*main(?!.*3fc)", r"TTT.*Main(?!.*3fc)"]
            elif combo == "ttt + robust":
                patterns = [r"ttt.*robust(?!.*3fc)", r"TTT.*Robust(?!.*3fc)"]
            elif combo == "ttt3fc + main":
                patterns = [r"ttt3fc.*main", r"TTT3fc.*Main"]
            elif combo == "ttt3fc + robust":
                patterns = [r"ttt3fc.*robust", r"TTT3fc.*Robust"]
            
            # Search for patterns
            found = False
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    found = True
                    break
            
            found_combinations[combo] = found
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
        
    return found_combinations

def check_early_stopping(filepath):
    """Check if file implements early stopping"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        patterns = [
            r"early.*stop",
            r"patience",
            r"best.*val",
            r"early_stopping",
            r"EarlyStopping"
        ]
        
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
        
    except:
        return False

# Files to check
files_to_check = [
    "main_baselines_3fc.py",
    "main_baselines_3fc_integration.py", 
    "main_cifar10_all.py"
]

print("="*80)
print("MODEL COMBINATION AND EARLY STOPPING CHECK")
print("="*80)

for file in files_to_check:
    if os.path.exists(file):
        print(f"\n📄 {file}:")
        print("-"*40)
        
        # Check early stopping
        has_early_stopping = check_early_stopping(file)
        print(f"✅ Early Stopping: {'YES' if has_early_stopping else 'NO'}")
        
        # Check model combinations
        combinations = check_file_for_combinations(file)
        if combinations:
            print("\nModel Combinations:")
            for combo, found in combinations.items():
                status = "✅" if found else "❌"
                print(f"  {status} {combo}")
                
            # Summary
            found_count = sum(1 for v in combinations.values() if v)
            total_count = len(combinations)
            print(f"\nFound {found_count}/{total_count} combinations")
    else:
        print(f"\n❌ {file} not found")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nRequired Model Combinations:")
print("1. Main model")
print("2. Main model robust")
print("3. BlendedTTT") 
print("4. BlendedTTT3fc")
print("5. Healer + Main")
print("6. Healer + Robust")
print("7. TTT + Main")
print("8. TTT + Robust")
print("9. TTT3fc + Main")
print("10. TTT3fc + Robust")