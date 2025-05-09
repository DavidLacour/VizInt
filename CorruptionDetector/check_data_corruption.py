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
    
    return class_counts

def check_for_class_zero():
    """
    Specifically check for the presence of class 0 (uncorrupted) images and
    print detailed information about them if they exist.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist")
        return False
    
    class_zero_images = []
    
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        
        if not os.path.isfile(file_path) or filename == "class_mapping.txt":
            continue
        
        if filename.startswith("uncorrupted_"):
            class_zero_images.append(filename)
    
    print("\nClass 0 (Uncorrupted) Verification:")
    print("=" * 50)
    
    if class_zero_images:
        print(f"Found {len(class_zero_images)} images for class 0 (uncorrupted)")
        print("\nSample of class 0 images:")
        for i, filename in enumerate(class_zero_images[:10], 1):  # Show first 10 examples
            file_size = os.path.getsize(os.path.join(dataset_path, filename)) / 1024  # Size in KB
            print(f"  {i}. {filename} ({file_size:.1f} KB)")
        
        if len(class_zero_images) > 10:
            print(f"  ... and {len(class_zero_images) - 10} more")
        
        return True
    else:
        print("ERROR: No images found for class 0 (uncorrupted)")
        print("Possible reasons:")
        print("  1. The 'normal_images_root' directory in dataset creation script doesn't exist")
        print("  2. The 'process_uncorrupted_images' function failed during dataset creation")
        print("  3. The image file extensions in the uncorrupted directory aren't recognized")
        print("\nRecommended actions:")
        print("  1. Check if the directory exists: ls -la /home/david-lacour/Documents/transformerVision/CorruptionDetector/ILSVRC2012_img_val/")
        print("  2. Run only the uncorrupted image processing part of the dataset creation script")
        print("  3. Check the logs from when you created the dataset for any warnings")
        
        return False

def verify_all_required_classes():
    """
    Verify that all classes from 0 to 25 are present in the dataset.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist")
        return
    
    # Load class mapping from file
    class_names = {}
    mapping_file = os.path.join(dataset_path, "class_mapping.txt")
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            for line in f:
                if line.startswith("Class "):
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        class_num = int(parts[0].replace("Class ", ""))
                        class_name = parts[1]
                        class_names[class_num] = class_name
    
    if not class_names:
        print("Warning: Could not load class mapping from file")
        expected_classes = set(range(26))  # Classes 0-25
    else:
        expected_classes = set(class_names.keys())
    
    # Initialize counters for each class
    class_counts = {i: 0 for i in expected_classes}
    
    # Count files for each class
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        if not os.path.isfile(file_path) or filename == "class_mapping.txt":
            continue
        
        if filename.startswith("uncorrupted_"):
            class_counts[0] += 1
        else:
            match = re.search(r'_(\d+)_', filename)
            if match:
                class_num = int(match.group(1))
                if class_num in class_counts:
                    class_counts[class_num] += 1
    
    # Check if any classes are missing
    missing_classes = [class_num for class_num in expected_classes if class_counts[class_num] == 0]
    
    print("\nClass Presence Verification:")
    print("=" * 50)
    
    if missing_classes:
        print(f"WARNING: {len(missing_classes)} classes are completely missing from the dataset:")
        for class_num in sorted(missing_classes):
            print(f"  Class {class_num}: {class_names.get(class_num, 'Unknown')} - 0 images")
    else:
        print("SUCCESS: All required classes (0-25) are present in the dataset")
    
    # Check for low count classes
    low_count_classes = [class_num for class_num in expected_classes 
                        if 0 < class_counts[class_num] < 20]  # Arbitrary threshold of 20
    
    if low_count_classes:
        print(f"\nWARNING: {len(low_count_classes)} classes have very few images (less than 20):")
        for class_num in sorted(low_count_classes):
            print(f"  Class {class_num}: {class_names.get(class_num, 'Unknown')} - {class_counts[class_num]} images")
    
    print("\nClass Counts Summary:")
    for class_num in sorted(expected_classes):
        class_name = class_names.get(class_num, 'Unknown')
        count = class_counts[class_num]
        status = "✓" if count >= 20 else "⚠" if count > 0 else "✗"
        print(f"  {status} Class {class_num}: {class_name} - {count} images")

if __name__ == "__main__":
    print("Checking filename uniqueness in the dataset...")
    duplicates = check_filename_uniqueness()
    
    print("\nAnalyzing class distribution...")
    class_counts = analyze_class_distribution()
    
    print("\nSpecifically checking for class 0 (uncorrupted) images...")
    has_class_zero = check_for_class_zero()
    
    print("\nVerifying all required classes...")
    verify_all_required_classes()
    
    if duplicates:
        print("\nSuggestion: Run the dataset creation script again with a different random seed or modify it to ensure each base image is used only once.")
    
    if not has_class_zero:
        print("\nCRITICAL: Your dataset is missing the 'uncorrupted' class (class 0).")
        print("This explains why this class is missing from your model's confusion matrix.")
        print("Run the dataset creation script again and ensure the process_uncorrupted_images function completes successfully.")