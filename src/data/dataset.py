import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import hydra
from omegaconf import DictConfig
from typing import Tuple, Dict, List, Optional, Any
from PIL import Image
import pandas as pd
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from src.utils.file_finder import find_image_path


class NIHChestXrayDataset(Dataset):
    """PyTorch Dataset for the NIH Chest X-ray dataset."""

    def __init__(self, filenames: Optional[List[str]] = None, 
                 labels: Optional[np.ndarray] = None, 
                 class_map: Dict[str, int] = None, 
                 data_root: str = None, 
                 transform = None,
                 data_list: Optional[List[Dict[str, Any]]] = None):
        """
        Args:
            filenames (List[str], optional): List of image filenames (Standard mode).
            labels (np.ndarray, optional): Array of labels (Standard mode).
            class_map (Dict[str, int]): Mapping from class name to integer label.
            data_root (str): Root directory for data.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_list (List[Dict], optional): Pre-loaded data from pickle (Cached mode).
                                            Each item is {'image': np.uint8, 'label': np.float32}.
        """
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
        self.class_map = class_map
        self.data_root = data_root
        self.data_list = data_list

    def __len__(self) -> int:
        if self.data_list is not None:
            return len(self.data_list)
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- Cached Mode (Pickle) ---
        if self.data_list is not None:
            item = self.data_list[idx]
            # Convert uint8 numpy array back to PIL Image
            image = Image.fromarray(item['image'])
            # Label is already in the item
            label = torch.tensor(item['label'], dtype=torch.float32)
            
            if self.transform:
                image = self.transform(image)
                
            return image, label

        # --- Standard Mode (File I/O) ---
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

def get_data_splits(cfg: DictConfig, debug: bool = False):
    """
    Loads data lists and splits them into train, val, test.
    Returns:
        splits (Dict): {'train': (files, labels), 'val': ..., 'test': ...}
        class_map (Dict): mapping
        metadata (pd.DataFrame): metadata df
    """
    data_root = hydra.utils.to_absolute_path(cfg.dataset.data_dir)
    target_classes = list(cfg.dataset.classes)
    class_map = {name: i for i, name in enumerate(target_classes)}
    num_classes = len(target_classes)

    # Load list files
    train_val_list_path = os.path.join(data_root, 'train_val_list.txt')
    test_list_path = os.path.join(data_root, 'test_list.txt')
    metadata_path = os.path.join(data_root, 'Data_Entry_2017.csv')

    print(f"Data root: {os.path.abspath(data_root)}")
    print(f"Checking for train_val_list: {train_val_list_path} (exists: {os.path.exists(train_val_list_path)})")

    def load_list(path, limit=None):
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            return []
        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(lines)} lines from {path}")
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
            # Add full image paths to metadata
            metadata['image_path'] = metadata['Image Index'].apply(lambda x: find_image_path(x, data_root))
            
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
        
    splits = {
        'train': (train_files, train_y),
        'val': (val_files, val_y),
        'test': (test_filenames, test_labels)
    }
    return splits, class_map, metadata

def get_dataloaders(cfg: DictConfig, debug: bool = False) -> Tuple[Dict[str, DataLoader], Optional[pd.DataFrame]]:
    """
    Creates and returns train, validation, and test dataloaders for the NIH dataset.
    Supports both standard file I/O and cached Pickle I/O.
    """
    data_root = hydra.utils.to_absolute_path(cfg.dataset.data_dir)
    target_classes = list(cfg.dataset.classes)
    class_map = {name: i for i, name in enumerate(target_classes)}

    # Define transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    data_transforms = {
        'train': transforms.Compose([
            # Resize is assumed to be done during pickle creation if using pickle
            # Input is already PIL Image in both modes (standard load or pickle load)
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    
    # Adjust transforms slightly logic:
    # If using pickle, input is PIL Image (converted in __getitem__).
    # If using standard, input is PIL Image.
    # So transforms are compatible. 
    # BUT, standard mode has `Resize` at the beginning. Pickle mode already has Resized data.
    # We should ensure we don't Resize again if not needed, or it doesn't hurt.
    # To be safe, let's keep it clean.
    
    if cfg.dataset.get('use_pickle', False):
        print(f"Loading data from Pickle files (Directory: {cfg.dataset.pickle_dir})...")
        pickle_dir = os.path.join(data_root, cfg.dataset.pickle_dir)
        datasets = {}
        
        for phase in ['train', 'val', 'test']:
            pkl_path = os.path.join(pickle_dir, f"{phase}.pkl")
            if not os.path.exists(pkl_path):
                print(f"Warning: Pickle file not found: {pkl_path}")
                continue
            
            # For debug mode, we might want to load only a subset, but pickle loading is all-or-nothing usually
            # unless we implement a custom lazy loader. For now, load all.
            with open(pkl_path, 'rb') as f:
                data_list = pickle.load(f)
                
            if debug:
                data_list = data_list[:10]
                
            print(f"Loaded {len(data_list)} samples for {phase}")
            
            # Select transform
            transform = data_transforms['train'] if phase == 'train' else data_transforms['val']
            
            datasets[phase] = NIHChestXrayDataset(
                class_map=class_map,
                data_root=data_root,
                transform=transform,
                data_list=data_list
            )
            
        metadata = None # Metadata is not easily available in pickle mode without loading CSV separately
        
    else:
        # Standard Mode
        splits, _, metadata = get_data_splits(cfg, debug)
        
        # Add Resize to transforms for Standard Mode
        # Note: We reconstruct transforms here to add Resize
        base_train_transforms = [
            transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        base_val_transforms = [
            transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        
        datasets = {}
        for phase, (files, labels) in splits.items():
            if len(files) > 0:
                tf = transforms.Compose(base_train_transforms if phase == 'train' else base_val_transforms)
                datasets[phase] = NIHChestXrayDataset(
                    filenames=files,
                    labels=labels,
                    class_map=class_map,
                    data_root=data_root,
                    transform=tf
                )

    dataloaders = {
        p: DataLoader(ds, batch_size=cfg.dataset.batch_size if p != 'train' or not debug else 2, 
                     shuffle=(p == 'train'), num_workers=cfg.dataset.num_workers) 
        for p, ds in datasets.items()
    }

    return dataloaders, metadata
