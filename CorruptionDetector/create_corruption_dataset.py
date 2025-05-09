import os
import shutil
import random
import re
from collections import defaultdict
from pathlib import Path

path = "/home/david-lacour/Documents/transformerVision/CorruptionDetector"  

imagenet_c_root = path + "/imagenet-c/"
laion_c_root = path + "/laion-c/" 
normal_images_root = path + "/uncorrupted/ILSVRC2012_img_val/"  
output_root = path + "/data_corruption_classifications/"

os.makedirs(output_root, exist_ok=True)

IMAGES_PER_CLASS = 100

class_numbers = {
    "uncorrupted": 0,
    "gaussian_noise": 1,
    "shot_noise": 2,
    "impulse_noise": 3,
    "defocus_blur": 4,
    "glass_blur": 5,
    "motion_blur": 6,
    "zoom_blur": 7,
    "frost": 8,
    "snow": 9,
    "fog": 10,
    "brightness": 11,
    "contrast": 12,
    "elastic_transform": 13,
    "pixelate": 14,
    "jpeg_compression": 15,
    "speckle_noise": 16,
    "spatter": 17,
    "gaussian_blur": 18,
    "saturate": 19,
    "geometric_shapes": 20,
    "glitched": 21,
    "luminance_checkerboard": 22,
    "mosaic": 23,
    "stickers": 24,
    "vertical_lines": 25
}

def extract_base_image_name(filename):
    match = re.search(r'(ILSVRC2012_\w+_\d+)', filename)
    if match:
        return match.group(1)
    
    return os.path.basename(filename)

def get_unique_filename(base_filename, used_filenames):
    new_filename = base_filename
    counter = 1
    base_name, ext = os.path.splitext(base_filename)
    
    while new_filename in used_filenames:
        new_filename = f"{base_name}_{counter}{ext}"
        counter += 1
        
    used_filenames.add(new_filename)
    return new_filename

def check_existing_files_in_output():
    existing_files = set()
    
    if os.path.exists(output_root):
        for file in os.listdir(output_root):
            if os.path.isfile(os.path.join(output_root, file)):
                existing_files.add(file)
    
    print(f"Found {len(existing_files)} existing files in output directory")
    return existing_files

def process_uncorrupted_images(used_filenames, used_base_image_names):
    class_num = class_numbers["uncorrupted"]
    
    uncorrupted_images = []
    
    try:
        if not os.path.exists(normal_images_root):
            print(f"Warning: Uncorrupted images path not found - {normal_images_root}")
            return
            
        for img_file in os.listdir(normal_images_root):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                continue
                
            src_path = os.path.join(normal_images_root, img_file)
            
            base_image_name = extract_base_image_name(src_path)
            
            if base_image_name in used_base_image_names:
                continue
                
            uncorrupted_images.append({
                "src_path": src_path,
                "img_file": img_file,
                "base_image_name": base_image_name
            })
    except Exception as e:
        print(f"Error processing uncorrupted images: {e}")
        return
    
    if len(uncorrupted_images) < IMAGES_PER_CLASS:
        print(f"Warning: Only {len(uncorrupted_images)} uncorrupted images available")
        selected_images = uncorrupted_images
    else:
        selected_images = random.sample(uncorrupted_images, IMAGES_PER_CLASS)
    
    copied_count = 0
    for img_data in selected_images:
        used_base_image_names.add(img_data["base_image_name"])
        
        base_filename = f"uncorrupted_{class_num}_{img_data['img_file']}"
        
        new_filename = get_unique_filename(base_filename, used_filenames)
        
        dst_path = os.path.join(output_root, new_filename)
        shutil.copy2(img_data["src_path"], dst_path)
        
        copied_count += 1
        print(f"Copied {img_data['src_path']} to {dst_path}")
    
    print(f"Added {copied_count} uncorrupted images (class {class_num})")

def process_imagenet_c(used_filenames, used_base_image_names):
    corruptions = {
        "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
        "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
        "weather": ["frost", "snow", "fog", "brightness"],
        "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
        "extra": ["speckle_noise", "spatter", "gaussian_blur", "saturate"]
    }
    
    for category, classes in corruptions.items():
        for corruption_class in classes:
            class_num = class_numbers.get(corruption_class, 99)
            
            all_images = []
            corruption_path = os.path.join(imagenet_c_root, category, corruption_class)
            
            intensity_levels = []
            try:
                intensity_levels = [d for d in os.listdir(corruption_path) if os.path.isdir(os.path.join(corruption_path, d))]
            except FileNotFoundError:
                print(f"Warning: Path not found - {corruption_path}")
                continue
            
            for level in intensity_levels:
                level_path = os.path.join(corruption_path, level)
                
                for img_class in os.listdir(level_path):
                    img_class_path = os.path.join(level_path, img_class)
                    
                    if not os.path.isdir(img_class_path):
                        continue
                        
                    for img_file in os.listdir(img_class_path):
                        src_path = os.path.join(img_class_path, img_file)
                        
                        base_image_name = extract_base_image_name(src_path)
                        
                        if base_image_name in used_base_image_names:
                            continue
                            
                        all_images.append({
                            "src_path": src_path,
                            "img_class": img_class,
                            "img_file": img_file,
                            "base_image_name": base_image_name,
                            "level": level
                        })
            
            unique_images_by_base_name = {}
            
            for img_data in all_images:
                if img_data["base_image_name"] not in unique_images_by_base_name:
                    unique_images_by_base_name[img_data["base_image_name"]] = img_data
            
            unique_images = list(unique_images_by_base_name.values())
            
            print(f"Found {len(unique_images)} unique images for {corruption_class} after deduplication")
            
            if len(unique_images) < IMAGES_PER_CLASS:
                print(f"Warning: Only {len(unique_images)} unique images available for {corruption_class}")
                selected_images = unique_images
            else:
                images_by_level = defaultdict(list)
                for img_data in unique_images:
                    images_by_level[img_data["level"]].append(img_data)
                
                images_per_level = IMAGES_PER_CLASS // len(images_by_level)
                extra_images = IMAGES_PER_CLASS % len(images_by_level)
                
                selected_images = []
                for level, images in images_by_level.items():
                    level_count = images_per_level + (1 if extra_images > 0 else 0)
                    extra_images -= 1 if extra_images > 0 else 0
                    
                    if len(images) < level_count:
                        selected_level_images = images
                    else:
                        selected_level_images = random.sample(images, level_count)
                    
                    selected_images.extend(selected_level_images)
                
                if len(selected_images) < IMAGES_PER_CLASS:
                    remaining = IMAGES_PER_CLASS - len(selected_images)
                    all_remaining = []
                    for level, images in images_by_level.items():
                        for img in images:
                            if img not in selected_images:
                                all_remaining.append(img)
                    
                    if len(all_remaining) >= remaining:
                        selected_images.extend(random.sample(all_remaining, remaining))
            
            copied_count = 0
            for img_data in selected_images:
                if img_data["base_image_name"] in used_base_image_names:
                    print(f"Warning: Base image {img_data['base_image_name']} was already used. Skipping.")
                    continue
                
                used_base_image_names.add(img_data["base_image_name"])
                
                base_filename = f"imagenet_{corruption_class}_{class_num}_L{img_data['level']}_{img_data['img_class']}_{img_data['img_file']}"
                
                new_filename = get_unique_filename(base_filename, used_filenames)
                
                dst_path = os.path.join(output_root, new_filename)
                shutil.copy2(img_data["src_path"], dst_path)
                
                copied_count += 1
                print(f"Copied {img_data['src_path']} to {dst_path}")
                
                if copied_count >= IMAGES_PER_CLASS:
                    break
            
            print(f"Added {copied_count} images for {corruption_class} (class {class_num})")

def process_laion_c(used_filenames, used_base_image_names):
    corruption_classes = [
        "geometric_shapes", "glitched", "luminance_checkerboard", 
        "mosaic", "stickers", "vertical_lines"
    ]
    
    for corruption_class in corruption_classes:
        class_num = class_numbers.get(corruption_class, 99)
        
        corruption_path = os.path.join(laion_c_root, corruption_class, corruption_class)
        
        try:
            if not os.path.exists(corruption_path):
                print(f"Warning: Path not found - {corruption_path}")
                continue
        except:
            print(f"Warning: Error accessing path - {corruption_path}")
            continue
        
        all_images = []
        
        intensity_levels = []
        try:
            intensity_levels = [d for d in os.listdir(corruption_path) 
                              if d.startswith("intensity_level_") and 
                              os.path.isdir(os.path.join(corruption_path, d))]
        except FileNotFoundError:
            print(f"Warning: No intensity levels found in {corruption_path}")
            continue
            
        for level_dir in intensity_levels:
            level = level_dir.split("_")[-1]
            level_path = os.path.join(corruption_path, level_dir)
            
            for obj_category in os.listdir(level_path):
                obj_path = os.path.join(level_path, obj_category)
                
                if not os.path.isdir(obj_path):
                    continue
                    
                for img_file in os.listdir(obj_path):
                    src_path = os.path.join(obj_path, img_file)
                    
                    base_image_name = extract_base_image_name(src_path)
                    
                    if base_image_name in used_base_image_names:
                        continue
                        
                    all_images.append({
                        "src_path": src_path,
                        "obj_category": obj_category,
                        "img_file": img_file,
                        "base_image_name": base_image_name,
                        "level": level
                    })
        
        unique_images_by_base_name = {}
        
        for img_data in all_images:
            if img_data["base_image_name"] not in unique_images_by_base_name:
                unique_images_by_base_name[img_data["base_image_name"]] = img_data
        
        unique_images = list(unique_images_by_base_name.values())
        
        print(f"Found {len(unique_images)} unique images for {corruption_class} after deduplication")
        
        if len(unique_images) < IMAGES_PER_CLASS:
            print(f"Warning: Only {len(unique_images)} unique images available for {corruption_class}")
            selected_images = unique_images
        else:
            images_by_level = defaultdict(list)
            for img_data in unique_images:
                images_by_level[img_data["level"]].append(img_data)
            
            images_per_level = IMAGES_PER_CLASS // len(images_by_level)
            extra_images = IMAGES_PER_CLASS % len(images_by_level)
            
            selected_images = []
            for level, images in images_by_level.items():
                level_count = images_per_level + (1 if extra_images > 0 else 0)
                extra_images -= 1 if extra_images > 0 else 0
                
                if len(images) < level_count:
                    selected_level_images = images
                else:
                    selected_level_images = random.sample(images, level_count)
                
                selected_images.extend(selected_level_images)
            
            if len(selected_images) < IMAGES_PER_CLASS:
                remaining = IMAGES_PER_CLASS - len(selected_images)
                all_remaining = []
                for level, images in images_by_level.items():
                    for img in images:
                        if img not in selected_images:
                            all_remaining.append(img)
                
                if len(all_remaining) >= remaining:
                    selected_images.extend(random.sample(all_remaining, remaining))
        
        copied_count = 0
        for img_data in selected_images:
            if img_data["base_image_name"] in used_base_image_names:
                print(f"Warning: Base image {img_data['base_image_name']} was already used. Skipping.")
                continue
            
            used_base_image_names.add(img_data["base_image_name"])
            
            base_filename = f"laion_{corruption_class}_{class_num}_L{img_data['level']}_{img_data['obj_category']}_{img_data['img_file']}"
            
            new_filename = get_unique_filename(base_filename, used_filenames)
            
            dst_path = os.path.join(output_root, new_filename)
            shutil.copy2(img_data["src_path"], dst_path)
            
            copied_count += 1
            print(f"Copied {img_data['src_path']} to {dst_path}")
            
            if copied_count >= IMAGES_PER_CLASS:
                break
        
        print(f"Added {copied_count} images for {corruption_class} (class {class_num})")

def save_class_mapping():
    with open(os.path.join(output_root, "class_mapping.txt"), "w") as f:
        f.write("Class Mapping for Corruption Classification Dataset\n")
        f.write("=" * 50 + "\n\n")
        for class_name, class_num in sorted(class_numbers.items(), key=lambda x: x[1]):
            f.write(f"Class {class_num}: {class_name}\n")

def check_dataset_uniqueness():
    if not os.path.exists(output_root):
        print(f"Error: Dataset path '{output_root}' does not exist")
        return
    
    image_occurrences = defaultdict(list)
    
    total_files = 0
    
    for filename in os.listdir(output_root):
        file_path = os.path.join(output_root, filename)
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

if __name__ == "__main__":
    used_filenames = check_existing_files_in_output()
    
    used_base_image_names = set()
    
    process_uncorrupted_images(used_filenames, used_base_image_names)
    
    process_imagenet_c(used_filenames, used_base_image_names)
    process_laion_c(used_filenames, used_base_image_names)
    
    save_class_mapping()
    
    print("Balanced dataset creation completed!")
    print(f"Total unique base images used: {len(used_base_image_names)}")
    
    print("\nVerifying dataset uniqueness...")
    duplicates = check_dataset_uniqueness()
    
    if not duplicates:
        print("\nClass mapping:")
        for class_name, class_num in sorted(class_numbers.items(), key=lambda x: x[1]):
            print(f"  Class {class_num}: {class_name}")