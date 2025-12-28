import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from omegaconf import DictConfig
from typing import Tuple, Dict, List, Optional
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.file_finder import find_image_path


class NIHChestXrayDataset(Dataset):
    """PyTorch Dataset for the NIH Chest X-ray dataset."""

    def __init__(self, filenames: List[str], labels: Optional[np.ndarray], class_map: Dict[str, int], data_root: str, transform=None):
        """
        Args:
            filenames (List[str]): List of image filenames.
            labels (np.ndarray, optional): Array of labels corresponding to filenames.
            class_map (Dict[str, int]): Mapping from class name to integer label.
            data_root (str): Root directory for data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
        self.class_map = class_map
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]
        img_path = find_image_path(filename, self.data_root)
        
        if self.labels is not None:
            label = self.labels[idx]
        else:
            # Dummy label (zeros) if no labels provided
            label = np.zeros(len(self.class_map), dtype=np.float32)

        try:
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
            else:
                # If path not found or image doesn't exist, create a dummy image
                print(f"Warning: Image {filename} not found at {img_path}. Using dummy image.")
                image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        except Exception as e:
            print(f"Warning: Failed to load image at {img_path}. Error: {e}. Returning a dummy image.")
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(cfg: DictConfig, debug: bool = False) -> Tuple[Dict[str, DataLoader], Optional[pd.DataFrame]]:
    """
    Creates and returns train, validation, and test dataloaders for the NIH dataset.
    """
    data_root = cfg.dataset.data_dir
    target_classes = list(cfg.dataset.classes)
    class_map = {name: i for i, name in enumerate(target_classes)}
    num_classes = len(target_classes)

    # Load list files
    train_val_list_path = os.path.join(data_root, 'train_val_list.txt')
    test_list_path = os.path.join(data_root, 'test_list.txt')
    metadata_path = os.path.join(data_root, 'Data_Entry_2017.csv')

    def load_list(path, limit=None):
        if not os.path.exists(path):
            return []
        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines[:limit] if limit else lines

    if debug:
        train_val_filenames = load_list(train_val_list_path, limit=2)
        test_filenames = []
    else:
        train_val_filenames = load_list(train_val_list_path)
        test_filenames = load_list(test_list_path)

    # Load labels if available
    metadata = None
    train_val_labels = None
    test_labels = None

    if os.path.exists(metadata_path):
        try:
            metadata = pd.read_csv(metadata_path)
            # Create label matrix
            def get_label_vector(label_str):
                vec = np.zeros(num_classes, dtype=np.float32)
                for i, cls in enumerate(target_classes):
                    if cls in label_str:
                        vec[i] = 1.0
                return vec

            label_dict = dict(zip(metadata['Image Index'], metadata['Finding Labels'].apply(get_label_vector)))
            
            train_val_labels = np.array([label_dict.get(f, np.zeros(num_classes)) for f in train_val_filenames])
            test_labels = np.array([label_dict.get(f, np.zeros(num_classes)) for f in test_filenames])
        except Exception as e:
            print(f"Warning: Error loading metadata: {e}. Using dummy labels.")

    # Split train_val into train and val
    if not debug and len(train_val_filenames) > 0:
        train_files, val_files, train_y, val_y = train_test_split(
            train_val_filenames, train_val_labels, test_size=0.1, random_state=cfg.seed
        )
    else:
        train_files = train_val_filenames
        val_files = [] # For debug=True, we just use train
        train_y = train_val_labels
        val_y = None

    # Define transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # Create datasets
    datasets = {}
    if len(train_files) > 0:
        datasets['train'] = NIHChestXrayDataset(train_files, train_y, class_map, data_root, transform=data_transforms['train'])
    if len(val_files) > 0:
        datasets['val'] = NIHChestXrayDataset(val_files, val_y, class_map, data_root, transform=data_transforms['val'])
    if len(test_filenames) > 0:
        datasets['test'] = NIHChestXrayDataset(test_filenames, test_labels, class_map, data_root, transform=data_transforms['val'])

    dataloaders = {
        p: DataLoader(ds, batch_size=cfg.dataset.batch_size if p != 'train' or not debug else 2, 
                     shuffle=(p == 'train'), num_workers=cfg.dataset.num_workers) 
        for p, ds in datasets.items()
    }

    return dataloaders, metadata
