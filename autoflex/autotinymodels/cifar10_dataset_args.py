"""
Add these modifications to main_cifar10_all.py to support dataset path arguments
"""

# 1. Add these arguments to the argparse section (around line 884):
"""
# Dataset options
parser.add_argument("--dataset", type=str, default="../../cifar10", 
                    help="Path to CIFAR-10 dataset (default: ../../cifar10)")
parser.add_argument("--checkpoint_dir", type=str, default="../../cifar10checkpoints",
                    help="Directory to save model checkpoints (default: ../../cifar10checkpoints)")
"""

# 2. Replace the hardcoded paths at the top of the file (lines 34-35) with:
"""
# Dataset configuration - will be overridden by command line args
DEFAULT_DATASET_PATH = "../../cifar10"
DEFAULT_CHECKPOINT_PATH = "../../cifar10checkpoints"
NUM_CLASSES = 10  # CIFAR-10 has 10 classes
IMG_SIZE = 32  # CIFAR-10 images are 32x32
"""

# 3. In the main() function, after args = parser.parse_args(), add:
"""
# Override paths with command line arguments
global DATASET_PATH, CHECKPOINT_PATH
DATASET_PATH = args.dataset
CHECKPOINT_PATH = args.checkpoint_dir

print(f"📁 Using dataset path: {DATASET_PATH}")
print(f"💾 Using checkpoint path: {CHECKPOINT_PATH}")

# Create checkpoint directory if it doesn't exist
create_directories()
"""

# 4. Or simpler approach - just modify these two lines at the beginning of main():
"""
# Override global paths if provided
if hasattr(args, 'dataset'):
    global DATASET_PATH
    DATASET_PATH = args.dataset
if hasattr(args, 'checkpoint_dir'):
    global CHECKPOINT_PATH  
    CHECKPOINT_PATH = args.checkpoint_dir
"""