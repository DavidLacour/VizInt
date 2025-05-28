import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, ood_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.ood_transform = ood_transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        with open(os.path.join(root_dir, 'wnids.txt'), 'r') as f:
            wnids = [line.strip() for line in f]
     
        for idx, wnid in enumerate(wnids):
            self.class_to_idx[wnid] = idx
       
        if split == 'train':
            for class_id, wnid in enumerate(wnids):
                img_dir = os.path.join(root_dir, 'train', wnid, 'images')
                if os.path.isdir(img_dir):
                    for img_file in os.listdir(img_dir):
                        if img_file.endswith('.JPEG'):
                            self.image_paths.append(os.path.join(img_dir, img_file))
                            self.labels.append(class_id)
        elif split == 'val':
            val_annotations_file = os.path.join(root_dir, 'val', 'val_annotations.txt')
            with open(val_annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    img_file, wnid = parts[0], parts[1]
                    if wnid in self.class_to_idx:
                        self.image_paths.append(os.path.join(root_dir, 'val', 'images', img_file))
                        self.labels.append(self.class_to_idx[wnid])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (64, 64), color='black')
        
        if self.ood_transform:
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(image)
            
            transformed_tensor, transform_params = self.ood_transform.apply_transforms(
                img_tensor, return_params=True
            )
            
            if self.transform:
                normalize_transform = None
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        normalize_transform = t
                        break
                
                if normalize_transform:
                    normalized_img = normalize_transform(img_tensor)
                    normalized_transformed = normalize_transform(transformed_tensor)
                    return normalized_img, normalized_transformed, label, transform_params
            
            return img_tensor, transformed_tensor, label, transform_params
        
        if self.transform:
            image = self.transform(image)
        
        return image, label