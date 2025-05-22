import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from new_new import ContinuousTransforms, TinyImageNetDataset


import os
import torch
import wandb
import numpy as np
import shutil
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from new_new import ContinuousTransforms, TinyImageNetDataset


import os
import torch
import wandb
import numpy as np
import shutil
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from copy import deepcopy

# Import our custom ViT model
from transformer_utils import set_seed, LayerNorm, Mlp, TransformerTrunk
from vit_implementation import create_vit_model, PatchEmbed, VisionTransformer


MAX_ROTATION = 360.0 
MAX_STD_GAUSSIAN_NOISE = 0.5
MAX_TRANSLATION_AFFINE = 0.1
MAX_SHEAR_ANGLE = 15.0
DEBUG = False 


def main_with_blended():
    # Set seed for reproducibility
    set_seed(42)
    
    # Dataset path
    dataset_path = "tiny-imagenet-200"
    
    if DEBUG:
        print("\n*** RUNNING IN DEBUG MODE WITH MINIMAL DATA ***\n")
    
    # Step 1: Check for existing main classification model or train a new one
    main_model_path = "bestmodel_main/best_model.pt"
    if os.path.exists(main_model_path):
        print(f"Found existing main classification model at {main_model_path}")
        # Initialize the model with the same architecture
        base_model = create_vit_model(
            img_size=64,
            patch_size=8,
            in_chans=3,
            num_classes=200,
            embed_dim=384,
            depth=8,
            head_dim=64,
            mlp_ratio=4.0,
            use_resnet_stem=True
        )
        # Load the saved weights - need to strip the "vit_model." prefix
        checkpoint = torch.load(main_model_path)
        
        # Create a new state dict with the correct keys
        new_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith("vit_model."):
                new_key = key[len("vit_model."):]
                new_state_dict[new_key] = value
        
        # Load the adjusted state dict
        base_model.load_state_dict(new_state_dict)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = base_model.to(device)
        base_model.eval()
        print(f"Loaded model with validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    else:
        print("=== Training Main Classification Model ===")
        base_model = train_main_model(dataset_path)
    
    # Step 2: Check for existing healer model or train a new one
    healer_model_path = "bestmodel_healer/best_model.pt"
    if os.path.exists(healer_model_path):
        print(f"Found existing healer model at {healer_model_path}")
        # Initialize the model with the same architecture
        healer_model = TransformationHealer(
            img_size=64,
            patch_size=8,
            in_chans=3,
            embed_dim=384,
            depth=6,
            head_dim=64
        )
        # Load the saved weights
        checkpoint = torch.load(healer_model_path)
        healer_model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        healer_model = healer_model.to(device)
        healer_model.eval()
        print(f"Loaded healer model with training loss: {checkpoint.get('train_loss', 'N/A')}")
    else:
        print("\n=== Training Transformation Healer Model ===")
        healer_model = train_healer_model(dataset_path, severity=1.0)
    
    # Step 3: Check for existing TTT model or train a new one
    ttt_model_path = "bestmodel_ttt/best_model.pt"
    if os.path.exists(ttt_model_path):
        print(f"Found existing TTT model at {ttt_model_path}")
        # Initialize the model with the base model
        ttt_model = TTTModel(base_model)
        # Load the saved weights
        checkpoint = torch.load(ttt_model_path)
        ttt_model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ttt_model = ttt_model.to(device)
        ttt_model.eval()
        print(f"Loaded TTT model with validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    else:
        print("\n=== Training Test-Time Training (TTT) Model ===")
        ttt_model = train_ttt_model(base_model, dataset_path)
    
    # Step 4: Check for existing BlendedTTT model or train a new one
    blended_model_path = "bestmodel_blended/best_model.pt"
    if os.path.exists(blended_model_path):
        print(f"Found existing BlendedTTT model at {blended_model_path}")
        # Initialize the model with the base model
        blended_model = BlendedTTT(base_model)
        # Load the saved weights
        checkpoint = torch.load(blended_model_path)
        blended_model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blended_model = blended_model.to(device)
        blended_model.eval()
        print(f"Loaded BlendedTTT model with validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    else:
        print("\n=== Training BlendedTTT Model ===")
        blended_model = train_blended_ttt_model(base_model, dataset_path)
    
    # Step 5: Comprehensive evaluation with all models
    print("\n=== Comprehensive Evaluation with BlendedTTT Model ===")
    
    # In debug mode, use a minimal set of severities
    if DEBUG:
        severities = [0.2]  # Just one severity level for debugging
        print("DEBUG MODE: Testing with just one severity level")
    else:
        severities = [0.2, 0.5, 0.8]  # Multiple severity levels for real testing
    
    all_results = evaluate_full_pipeline_with_blended(
        base_model, healer_model, ttt_model, blended_model, dataset_path, severities
    )
    
    print("\nExperiment completed!")


# Add a new entry point to run just the BlendedTTT part
if __name__ == "__main__":
    # Parse command line arguments to decide which model to run
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Vision Transformer OOD Experiments")
    parser.add_argument("--blended", action="store_true", help="Run with BlendedTTT model")
    
    args = parser.parse_args()
    
    if args.blended:
        # Run the version with BlendedTTT
        main_with_blended()
    else:
        # Run the original code
        main()