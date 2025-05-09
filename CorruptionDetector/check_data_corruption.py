import os
import re
from collections import defaultdict

dataset_path = "/home/david-lacour/Documents/transformerVision/CorruptionDetector/data_corruption_classifications/"

def extract_base_image_name(filename):
    if filename.startswith("uncorrupted_"):
        parts = filename.split('_')
        return '_'.join(parts[2:])
    
    elif filename.startswith("imagenet_"):
        match = re.search(r'(ILSVRC\d+_\w+_\d+)', filename)
        if match:
            return match.group(1)
    
    elif filename.startswith("laion_"):
        match = re.search(r'(ILSVRC\d+_\w+_\d+)', filename)
        if match:
            return match.group(1)
    
    return filename

def check_filename_uniqueness():
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist")
        return
    
    image_occurrences = defaultdict(list)
    
    total_files = 0
    
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        if not os.path.isfile(file_path) or filename == "class_mapping.txt":
            continue
        
        total_files += 1
        
        base_name = extract_base_image_name(filename)
        
        image_occurrences[base_name].append(filename)
    
    duplicates = {base_name: filenames for base_name, filenames in image_occurrences.items() 
                 if len(filenames) > 1}
    
    print(f"Total files analyzed: {total_files}")
    print(f"Unique base image names: {len(image_occurrences)}")
    
    if duplicates:
        print(f"Found {len(duplicates)} base image names used multiple times:")
        for base_name, filenames in duplicates.items():
            print(f"\nBase name: {base_name} is used {len(filenames)} times:")
            for i, filename in enumerate(filenames, 1):
                print(f"  {i}. {filename}")
    else:
        print("Success! Each base image name is used only once in the dataset.")
    
    return duplicates

def analyze_class_distribution():
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist")
        return
    
    class_counts = defaultdict(int)
    
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        if not os.path.isfile(file_path) or filename == "class_mapping.txt":
            continue
        
        if filename.startswith("uncorrupted_"):
            class_name = "uncorrupted"
            class_num = 0
        else:
            match = re.search(r'_(\d+)_', filename)
            if match:
                class_num = int(match.group(1))
                
                if filename.startswith("imagenet_"):
                    parts = filename.split('_')
                    class_name = parts[1]
                elif filename.startswith("laion_"):
                    parts = filename.split('_')
                    class_name = parts[1]
                else:
                    class_name = f"Unknown (Class {class_num})"
            else:
                class_name = "Unknown"
                class_num = -1
        
        class_counts[f"{class_name} (Class {class_num})"] += 1
    
    print("\nClass Distribution:")
    print("=" * 50)
    for class_name, count in sorted(class_counts.items(), key=lambda x: int(re.search(r'\(Class (\d+)\)', x[0]).group(1))):
        print(f"{class_name}: {count} images")
    
    expected_count = 100
    unbalanced_classes = {class_name: count for class_name, count in class_counts.items() 
                         if count != expected_count}
    
    if unbalanced_classes:
        print(f"\nWarning: Found {len(unbalanced_classes)} classes with unexpected counts (expected {expected_count}):")
        for class_name, count in unbalanced_classes.items():
            print(f"  {class_name}: {count} images")
    else:
        print(f"\nAll classes have exactly {expected_count} images as expected.")

if __name__ == "__main__":
    print("Checking filename uniqueness in the dataset...")
    duplicates = check_filename_uniqueness()
    
    print("\nAnalyzing class distribution...")
    analyze_class_distribution()
    
    if duplicates:
        print("\nSuggestion: Run the dataset creation script again with a different random seed or modify it to ensure each base image is used only once.")