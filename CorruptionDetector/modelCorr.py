import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


num_epochs = 9


def extract_class_number(filename):
    if filename.startswith("uncorrupted_"):
        return 0
    
    match = re.search(r'_(\d+)_', filename)
    if match:
        return int(match.group(1))
    
    return -1

class CorruptionDataset(Dataset):
    def __init__(self, img_dir, processor, transform=None, split='train', test_size=0.2, random_state=42):
        self.img_dir = img_dir
        self.processor = processor
        self.transform = transform
        
        all_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpeg', '.jpg', '.JPEG', '.JPG')) 
                      and os.path.isfile(os.path.join(img_dir, f))]
        
        all_classes = [extract_class_number(f) for f in all_images]
        
        try:
            train_images, test_images = train_test_split(
                all_images, test_size=test_size, random_state=random_state, stratify=all_classes
            )
        except ValueError as e:
            print(f"Warning: Could not perform stratified split: {e}")
            train_images, test_images = train_test_split(
                all_images, test_size=test_size, random_state=random_state
            )
        
        self.images = train_images if split == 'train' else test_images if split == 'test' else all_images
        
        print(f"Found {len(self.images)} images for {split} set in {img_dir}")
        
        self.class_counts = {}
        for img in self.images:
            class_num = extract_class_number(img)
            self.class_counts[class_num] = self.class_counts.get(class_num, 0) + 1
        
        print("Class distribution:")
        for class_num, count in sorted(self.class_counts.items()):
            print(f"  Class {class_num}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        label = extract_class_number(img_name)
        
        pixel_values = self.processor(images=image, return_tensors="pt")
        pixel_values = {k: v.squeeze() for k, v in pixel_values.items()}
        
        return {"pixel_values": pixel_values["pixel_values"], "labels": torch.tensor(label)}

def predict_image(image_path, model, processor, device):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    probabilities = nn.Softmax(dim=1)(logits)[0]
    confidence = probabilities[predicted_class].item()
    
    return predicted_class, confidence, probabilities.cpu().numpy()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    unique_classes = sorted(set(y_true) | set(y_pred))
    
    labels = [class_names.get(i, f"Class {i}") for i in unique_classes]
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

def main():
    data_dir = "/home/david-lacour/Documents/transformerVision/CorruptionDetector/data_corruption_classifications"
    output_dir = "/home/david-lacour/Documents/transformerVision/CorruptionDetector/corruption_classifier_model"
    
    num_classes = 26
    
    class_names = {}
    mapping_file = os.path.join(data_dir, "class_mapping.txt")
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
        class_names = {i: f"Class {i}" for i in range(num_classes)}
    
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    train_dataset = CorruptionDataset(img_dir=data_dir, processor=processor, split='train')
    test_dataset = CorruptionDataset(img_dir=data_dir, processor=processor, split='test')
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
   
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        
        model.eval()
        test_correct = 0
        test_total = 0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(pixel_values=pixel_values)
                predictions = outputs.logits.argmax(-1)
                
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
        
        test_accuracy = test_correct / test_total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
    
    print("\nFinal Evaluation:")
    
    unique_classes = sorted(set(y_true))
    
    target_names = [class_names.get(i, f"Class {i}") for i in unique_classes]
    
    print(classification_report(y_true, y_pred, labels=unique_classes, target_names=target_names))
    
    plot_confusion_matrix(
        y_true, 
        y_pred, 
        class_names,
        os.path.join(output_dir, "confusion_matrix.png")
    )
    
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    print("\nTesting model on example images:")
    model.eval()
    
    test_examples = {}
    for img_name in test_dataset.images:
        class_num = extract_class_number(img_name)
        if class_num not in test_examples:
            test_examples[class_num] = []
        if len(test_examples[class_num]) < 2:
            test_examples[class_num].append(img_name)
    
    for class_num, examples in test_examples.items():
        print(f"\nClass {class_num} ({class_names.get(class_num, f'Class {class_num}')}):")
        for img_name in examples:
            img_path = os.path.join(data_dir, img_name)
            predicted_class, confidence, _ = predict_image(img_path, model, processor, device)
            print(f"  Image: {img_name}")
            print(f"  Predicted: Class {predicted_class} ({class_names.get(predicted_class, f'Class {predicted_class}')}) with confidence: {confidence:.4f}")
            print(f"  Correct: {'✓' if predicted_class == class_num else '✗'}")

if __name__ == "__main__":
    main()