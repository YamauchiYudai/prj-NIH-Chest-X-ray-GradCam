import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from omegaconf import DictConfig
from typing import List, Dict, Any, Tuple
from PIL import Image
import pandas as pd
import pydicom
import numpy as np
import os


def dicom_to_pil(dicom_path: str) -> Image.Image:
    """Reads a DICOM file and converts it to a PIL Image."""
    dcm = pydicom.dcmread(dicom_path)
    
    # Correct for MONOCHROME1, where high values are black
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = np.amax(dcm.pixel_array) - dcm.pixel_array
    else:
        pixel_array = dcm.pixel_array

    # Normalize to 0-255 and convert to 8-bit unsigned integer
    pixel_array = pixel_array - np.min(pixel_array)
    pixel_array = pixel_array / np.max(pixel_array)
    pixel_array = (pixel_array * 255).astype(np.uint8)
    
    return Image.fromarray(pixel_array)


class VinDRCXRDataset(Dataset):
    """PyTorch Dataset for the VinDR-CXR dataset."""

    def __init__(self, df: pd.DataFrame, class_map: Dict[str, int], transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            class_map (Dict[str, int]): Mapping from class name to integer label.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.transform = transform
        self.class_map = class_map

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx, 0]
        label_name = self.df.iloc[idx, 1]
        
        try:
            image = dicom_to_pil(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to load DICOM at {img_path}. Error: {e}. Returning a dummy tensor.")
            return torch.zeros(3, 224, 224), -1

        label = self.class_map[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(cfg: DictConfig) -> Tuple[Dict[str, DataLoader], pd.DataFrame]:
    """
    Creates and returns train, validation, and test dataloaders, plus the training dataframe.
    """
    
    # This path needs to point to the root of the extracted PhysioNet data
    # e.g., /path/to/vindr-cxr/1.0.0/
    base_data_dir = cfg.dataset.data_dir
    annotations_dir = os.path.join(base_data_dir, 'annotations')
    
    train_meta_path = os.path.join(annotations_dir, 'image_labels_train.csv')
    test_meta_path = os.path.join(annotations_dir, 'image_labels_test.csv')

    try:
        train_meta = pd.read_csv(train_meta_path)
        test_meta = pd.read_csv(test_meta_path)
        
        # Construct the full path to the DICOM files
        train_meta['image_path'] = train_meta['image_id'].apply(
            lambda x: os.path.join(base_data_dir, 'train', f"{x}.dicom")
        )
        test_meta['image_path'] = test_meta['image_id'].apply(
            lambda x: os.path.join(base_data_dir, 'test', f"{x}.dicom")
        )

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Metadata CSV not found. Expected at: {e.filename}. "
            "Please ensure `dataset.data_dir` in `conf/dataset/vindr.yaml` "
            "points to the correct location of the downloaded and extracted '1.0.0' directory."
        )

    # Filter for the classes of interest
    target_classes = cfg.dataset.classes
    class_map = {name: i for i, name in enumerate(target_classes)}

    # This is a multi-label dataset. We will simplify to single-label for this prototype.
    # We'll take the first label found from our target list for each image.
    def filter_and_map(df: pd.DataFrame) -> pd.DataFrame:
        # Filter rows where 'label_name' is one of our target classes
        df_filtered = df[df['label_name'].isin(target_classes)].copy()
        # Drop duplicates to keep one entry per image_id, simplifying to single-label
        # Return this unfiltered version for the gradcam pathology search
        return df_filtered

    train_df_full = filter_and_map(train_meta)
    train_df = train_df_full.drop_duplicates(subset='image_id', keep='first')
    
    test_df_filtered = filter_and_map(test_meta)
    test_df = test_df_filtered.drop_duplicates(subset='image_id', keep='first')


    # Split the test set to create a validation set
    val_df = test_df.sample(frac=0.5, random_state=cfg.seed)
    test_df = test_df.drop(val_df.index)

    print(f"Dataset sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    if len(train_df) == 0:
        raise ValueError(
            "No training images found for the specified classes. "
            "Please check class names in `conf/dataset/vindr.yaml` and verify dataset files."
        )

    # Define transforms
    mean = [0.485, 0.456, 0.406] # Standard ImageNet stats
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
    train_dataset = VinDRCXRDataset(train_df, class_map, transform=data_transforms['train'])
    val_dataset = VinDRCXRDataset(val_df, class_map, transform=data_transforms['val'])
    test_dataset = VinDRCXRDataset(test_df, class_map, transform=data_transforms['val'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)
    }

    return dataloaders, train_df_full
