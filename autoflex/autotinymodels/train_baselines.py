#!/usr/bin/env python3
"""
Training script for baseline models (ResNet18 and VGG16)
Saves models to autotinymodels directory
"""

import os
import argparse
import torch
from baseline_models import SimpleResNet18, SimpleVGG16, train_baseline_model

def main():
    parser = argparse.ArgumentParser(description="Train baseline models for Tiny ImageNet")
    parser.add_argument("--dataset", type=str, default="../tiny-imagenet-200",
                        help="Path to the Tiny ImageNet dataset")
    parser.add_argument("--models", type=str, default="resnet18,vgg16",
                        help="Comma-separated list of models to train (resnet18,vgg16)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--only_evaluate", action="store_true",
                        help="Only evaluate existing models, don't train")
    args = parser.parse_args()
    
    # Parse models to train
    models_to_train = args.models.split(',')
    
    # Make sure we're in the autotinymodels directory
    save_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Models will be saved to: {save_dir}")
    
    # Train ResNet18 if requested
    if 'resnet18' in models_to_train:
        resnet_path = os.path.join(save_dir, "bestmodel_resnet18_baseline/best_model.pt")
        if os.path.exists(resnet_path) and args.only_evaluate:
            print(f"\n✅ ResNet18 model already exists at {resnet_path}")
        else:
            print("\n" + "="*80)
            print("Training ResNet18 baseline model...")
            print("="*80)
            model = SimpleResNet18(num_classes=200)
            trained_model = train_baseline_model(
                model, 
                args.dataset, 
                model_name="resnet18_baseline", 
                epochs=args.epochs, 
                lr=args.lr
            )
            print(f"✅ ResNet18 training completed. Model saved to {resnet_path}")
    
    # Train VGG16 if requested
    if 'vgg16' in models_to_train:
        vgg_path = os.path.join(save_dir, "bestmodel_vgg16_baseline/best_model.pt")
        if os.path.exists(vgg_path) and args.only_evaluate:
            print(f"\n✅ VGG16 model already exists at {vgg_path}")
        else:
            print("\n" + "="*80)
            print("Training VGG16 baseline model...")
            print("="*80)
            model = SimpleVGG16(num_classes=200)
            trained_model = train_baseline_model(
                model, 
                args.dataset, 
                model_name="vgg16_baseline", 
                epochs=args.epochs, 
                lr=args.lr
            )
            print(f"✅ VGG16 training completed. Model saved to {vgg_path}")
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    print("\nTrained models saved to:")
    if 'resnet18' in models_to_train:
        print(f"  - ResNet18: {save_dir}/bestmodel_resnet18_baseline/")
    if 'vgg16' in models_to_train:
        print(f"  - VGG16: {save_dir}/bestmodel_vgg16_baseline/")
    print("\nTo evaluate these models, use the evaluation script.")

if __name__ == "__main__":
    main()