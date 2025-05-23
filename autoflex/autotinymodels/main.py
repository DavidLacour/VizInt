import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path

from new_new import * 
# Import BlendedTTT modules
from blended_ttt_model import BlendedTTT
from blended_ttt_training import train_blended_ttt_model
from blended_ttt_evaluation import evaluate_models_with_blended, evaluate_full_pipeline_with_blended

def visualize_transformations(model_dir, dataset_path, num_samples=5, severity=0.5, save_dir="visualizations", include_blended=True):
    """
    Visualize original, transformed, and corrected images to illustrate the healing process.
    
    Args:
        model_dir: Directory containing the trained models
        dataset_path: Path to the dataset
        num_samples: Number of samples to visualize
        severity: Severity of transformations to apply
        save_dir: Directory to save visualizations
        include_blended: Whether to include BlendedTTT in visualizations
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    # Main classification model
    main_model = create_vit_model(
        img_size=64, patch_size=8, in_chans=3, num_classes=200,
        embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
    )
    main_checkpoint = torch.load(f"{model_dir}/bestmodel_main/best_model.pt", map_location=device)
    
    # Create a new state dict with the correct keys for the main model
    new_state_dict = {}
    for key, value in main_checkpoint['model_state_dict'].items():
        if key.startswith("vit_model."):
            new_key = key[len("vit_model."):]
            new_state_dict[new_key] = value
    
    main_model.load_state_dict(new_state_dict)
    main_model = main_model.to(device)
    main_model.eval()
    
    # Healer model
    healer_model = TransformationHealer(
        img_size=64, patch_size=8, in_chans=3,
        embed_dim=384, depth=6, head_dim=64
    )
    healer_checkpoint = torch.load(f"{model_dir}/bestmodel_healer/best_model.pt", map_location=device)
    healer_model.load_state_dict(healer_checkpoint['model_state_dict'])
    healer_model = healer_model.to(device)
    healer_model.eval()
    
    # BlendedTTT model (if requested)
    blended_model = None
    if include_blended:
        blended_model = BlendedTTT(
            base_model=main_model,
            img_size=64, 
            patch_size=8,
            embed_dim=384, 
            depth=4
        )
        blended_checkpoint = torch.load(f"{model_dir}/bestmodel_blended/best_model.pt", map_location=device)
        blended_model.load_state_dict(blended_checkpoint['model_state_dict'])
        blended_model = blended_model.to(device)
        blended_model.eval()
    
    # Setup transformations
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Initialize continuous transforms
    ood_transform = ContinuousTransforms(severity=severity)
    
    # Load dataset - use validation set for visualization
    val_dataset = TinyImageNetDataset(
        dataset_path, "val", transform_val, ood_transform=ood_transform
    )
    
    # For inverse normalization to display images
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])
    
    # Generate visualizations
    transform_names = ['no_transform', 'gaussian_noise', 'rotation', 'affine']
    
    with torch.no_grad():
        # Visualize each transformation type
        for t_idx, t_type in enumerate(transform_names):
            if t_type == 'no_transform':
                continue  # Skip no_transform for visualization
                
            # Determine number of columns based on whether BlendedTTT is included
            num_cols = 4 if include_blended else 3
            plt.figure(figsize=(5 * num_cols, 5 * num_samples))
            plt.suptitle(f"{t_type.upper()} Transformation (Severity: {severity})", fontsize=16)
            
            for i in range(num_samples):
                # Get a random sample
                idx = np.random.randint(0, len(val_dataset))
                orig_img, trans_img, label, params = val_dataset[idx]
                
                # Convert to tensors and move to device
                orig_img = orig_img.unsqueeze(0).to(device)
                
                # Apply specific transformation
                trans_img, params = ood_transform.apply_transforms(
                    orig_img.squeeze(0).cpu(), 
                    transform_type=t_type, 
                    severity=severity,
                    return_params=True
                )
                trans_img = trans_img.unsqueeze(0).to(device)
                
                # Predict transformations with healer
                healer_predictions = healer_model(trans_img)
                
                # Get prediction details
                pred_type_idx = torch.argmax(healer_predictions['transform_type_logits'], dim=1).item()
                pred_type = transform_names[pred_type_idx]
                
                # Apply correction with healer model
                healer_corrected_img = healer_model.apply_correction(trans_img, healer_predictions)
                
                # Apply correction with BlendedTTT if available
                blended_pred_type = None
                if include_blended and blended_model is not None:
                    _, blended_predictions = blended_model(trans_img)
                    blended_pred_type_idx = torch.argmax(blended_predictions['transform_type_logits'], dim=1).item()
                    blended_pred_type = transform_names[blended_pred_type_idx]
                    
                    # Get model predictions for classification
                    blended_pred = torch.argmax(blended_model(trans_img)[0], dim=1).item()
                
                # Get model predictions
                orig_pred = torch.argmax(main_model(orig_img), dim=1).item()
                trans_pred = torch.argmax(main_model(trans_img), dim=1).item()
                healer_corrected_pred = torch.argmax(main_model(healer_corrected_img), dim=1).item()
                
                # Prepare images for display by denormalizing
                orig_disp = inv_normalize(orig_img.squeeze(0).cpu())
                trans_disp = inv_normalize(trans_img.squeeze(0).cpu())
                healer_corrected_disp = inv_normalize(healer_corrected_img.squeeze(0).cpu())
                
                # Convert to PIL for display
                to_pil = transforms.ToPILImage()
                orig_pil = to_pil(torch.clamp(orig_disp, 0, 1))
                trans_pil = to_pil(torch.clamp(trans_disp, 0, 1))
                healer_corrected_pil = to_pil(torch.clamp(healer_corrected_disp, 0, 1))
                
                # Plot images
                if include_blended:
                    # With BlendedTTT (4 columns: original, transformed, healer, blended)
                    plt.subplot(num_samples, num_cols, i*num_cols + 1)
                    plt.imshow(orig_pil)
                    plt.title(f"Original\nPrediction: {orig_pred}")
                    plt.axis('off')
                    
                    plt.subplot(num_samples, num_cols, i*num_cols + 2)
                    plt.imshow(trans_pil)
                    plt.title(f"Transformed ({t_type})\nPrediction: {trans_pred}")
                    plt.axis('off')
                    
                    plt.subplot(num_samples, num_cols, i*num_cols + 3)
                    plt.imshow(healer_corrected_pil)
                    plt.title(f"Healer Corrected (Detected: {pred_type})\nPrediction: {healer_corrected_pred}")
                    plt.axis('off')
                    
                    plt.subplot(num_samples, num_cols, i*num_cols + 4)
                    plt.imshow(trans_pil)  # For BlendedTTT we just show the classification result
                    plt.title(f"BlendedTTT (Detected: {blended_pred_type})\nPrediction: {blended_pred}")
                    plt.axis('off')
                else:
                    # Without BlendedTTT (original 3 columns)
                    plt.subplot(num_samples, num_cols, i*num_cols + 1)
                    plt.imshow(orig_pil)
                    plt.title(f"Original\nPrediction: {orig_pred}")
                    plt.axis('off')
                    
                    plt.subplot(num_samples, num_cols, i*num_cols + 2)
                    plt.imshow(trans_pil)
                    plt.title(f"Transformed ({t_type})\nPrediction: {trans_pred}")
                    plt.axis('off')
                    
                    plt.subplot(num_samples, num_cols, i*num_cols + 3)
                    plt.imshow(healer_corrected_pil)
                    plt.title(f"Corrected (Detected: {pred_type})\nPrediction: {healer_corrected_pred}")
                    plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{t_type}_visualization.png")
            plt.close()
            
            print(f"Saved visualization for {t_type} transformation")

def main():
    parser = argparse.ArgumentParser(description="Transform Healing with Vision Transformers")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "visualize", "all"],
                      help="Mode of operation: train, evaluate, visualize, or all")
    parser.add_argument("--dataset", type=str, default="tiny-imagenet-200",
                      help="Path to the dataset")
    parser.add_argument("--model_dir", type=str, default="./",
                      help="Directory to save/load models")
    parser.add_argument("--visualize_dir", type=str, default="visualizations",
                      help="Directory to save visualizations")
    parser.add_argument("--severity", type=float, default=0.5,
                      help="Severity of transformations for evaluation/visualization")
    parser.add_argument("--num_samples", type=int, default=5,
                      help="Number of samples to visualize")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--include_blended", action="store_true",
                      help="Whether to include BlendedTTT model in the pipeline")
    parser.add_argument("--skip_ttt", action="store_true",
                      help="Skip training TTT model (only relevant when --include_blended is used)")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    main_model = None
    healer_model = None
    ttt_model = None
    blended_model = None
    
    # Training mode
    if args.mode in ["train", "all"]:
        print("\n=== Training Main Classification Model ===")
        main_model = train_main_model(args.dataset)
        
        print("\n=== Training Transformation Healer Model ===")
        healer_model = train_healer_model(args.dataset, severity=args.severity)
        
        # Train TTT model if not skipped (when BlendedTTT is used, TTT might be redundant)
        if not args.skip_ttt:
            print("\n=== Training Test-Time Training Model ===")
            ttt_model = train_ttt_model(args.dataset, severity=args.severity)
        
        # Train BlendedTTT model if requested
        if args.include_blended:
            print("\n=== Training BlendedTTT Model ===")
            blended_model = train_blended_ttt_model(main_model, args.dataset)
    
    # Evaluation mode
    if args.mode in ["evaluate", "all"]:
        if main_model is None or healer_model is None:
            print("Loading models for evaluation...")
            
            # Load main model if not already loaded
            if main_model is None:
                main_model_path = f"{args.model_dir}/bestmodel_main/best_model.pt"
                if os.path.exists(main_model_path):
                    print(f"Loading main model from {main_model_path}")
                    main_model = create_vit_model(
                        img_size=64, patch_size=8, in_chans=3, num_classes=200,
                        embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
                    )
                    checkpoint = torch.load(main_model_path, map_location=device)
                    
                    # Create a new state dict with the correct keys
                    new_state_dict = {}
                    for key, value in checkpoint['model_state_dict'].items():
                        if key.startswith("vit_model."):
                            new_key = key[len("vit_model."):]
                            new_state_dict[new_key] = value
                    
                    main_model.load_state_dict(new_state_dict)
                    main_model = main_model.to(device)
                    main_model.eval()
                else:
                    print(f"Error: Main model not found at {main_model_path}")
                    if args.mode == "evaluate":
                        return
            
            # Load healer model if not already loaded
            if healer_model is None:
                healer_model_path = f"{args.model_dir}/bestmodel_healer/best_model.pt"
                if os.path.exists(healer_model_path):
                    print(f"Loading healer model from {healer_model_path}")
                    healer_model = TransformationHealer(
                        img_size=64, patch_size=8, in_chans=3,
                        embed_dim=384, depth=6, head_dim=64
                    )
                    checkpoint = torch.load(healer_model_path, map_location=device)
                    healer_model.load_state_dict(checkpoint['model_state_dict'])
                    healer_model = healer_model.to(device)
                    healer_model.eval()
                else:
                    print(f"Error: Healer model not found at {healer_model_path}")
                    if args.mode == "evaluate":
                        return
            
            # Load TTT model if applicable
            if not args.skip_ttt:
                ttt_model_path = f"{args.model_dir}/bestmodel_ttt/best_model.pt"
                if os.path.exists(ttt_model_path):
                    print(f"Loading TTT model from {ttt_model_path}")
                    ttt_model = TestTimeTrainer(
                        base_model=main_model,
                        img_size=64,
                        patch_size=8,
                        embed_dim=384
                    )
                    checkpoint = torch.load(ttt_model_path, map_location=device)
                    ttt_model.load_state_dict(checkpoint['model_state_dict'])
                    ttt_model = ttt_model.to(device)
                    ttt_model.eval()
                else:
                    print(f"Warning: TTT model not found at {ttt_model_path}")
            
            # Load BlendedTTT model if requested
            if args.include_blended:
                blended_model_path = f"{args.model_dir}/bestmodel_blended/best_model.pt"
                if os.path.exists(blended_model_path):
                    print(f"Loading BlendedTTT model from {blended_model_path}")
                    blended_model = BlendedTTT(
                        base_model=main_model,
                        img_size=64,
                        patch_size=8,
                        embed_dim=384,
                        depth=4
                    )
                    checkpoint = torch.load(blended_model_path, map_location=device)
                    blended_model.load_state_dict(checkpoint['model_state_dict'])
                    blended_model = blended_model.to(device)
                    blended_model.eval()
                else:
                    print(f"Warning: BlendedTTT model not found at {blended_model_path}")
        
        print("\n=== Comprehensive Evaluation ===")
        # Evaluate at multiple severity levels
        severities = [0.2, 0.5, 0.8]
        
        if args.include_blended:
            # Use integrated evaluation with BlendedTTT
            all_results = evaluate_full_pipeline_with_blended(
                main_model, healer_model, ttt_model, blended_model, 
                args.dataset, severities
            )
        else:
            # Use standard evaluation
            all_results = evaluate_full_pipeline(
                main_model, healer_model, ttt_model, 
                args.dataset, severities
            )
        
    # Visualization mode
    if args.mode in ["visualize", "all"]:
        print("\n=== Generating Visualizations ===")
        visualize_transformations(
            model_dir=args.model_dir,
            dataset_path=args.dataset,
            num_samples=args.num_samples,
            severity=args.severity,
            save_dir=args.visualize_dir,
            include_blended=args.include_blended
        )
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main()
