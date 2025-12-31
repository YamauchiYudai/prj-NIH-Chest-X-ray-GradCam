import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import torch

from src.data.dataset import get_data_splits, NIHChestXrayDataset

# Register OmegaConf resolvers
try:
    OmegaConf.register_new_resolver("len", len)
    OmegaConf.register_new_resolver("to_int", int)
except ValueError:
    pass # Already registered

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("--- Creating Pickle Files ---")
    
    # 1. Get data splits using the refactored function
    splits, class_map, _ = get_data_splits(cfg)
    data_root = hydra.utils.to_absolute_path(cfg.dataset.data_dir)
    
    # Define output dir
    output_dir = os.path.join(data_root, "pickles")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define transform: Resize only
    # We want to keep it as uint8 PIL image initially, then convert to numpy
    resize_transform = transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size))
    
    for split_name, (files, labels) in splits.items():
        if len(files) == 0:
            print(f"Skipping {split_name} (empty)")
            continue
            
        print(f"Processing {split_name} split ({len(files)} images)...")
        
        # Instantiate dataset to reuse path finding logic
        # We pass None as transform so we get the PIL image
        dataset = NIHChestXrayDataset(files, labels, class_map, data_root, transform=None)
        
        data_list = []
        
        for i in tqdm(range(len(dataset))):
            try:
                # __getitem__ returns (PIL.Image, torch.Tensor) when transform is None
                image, label = dataset[i]
                
                # Apply resize
                image = resize_transform(image)
                
                # Convert to numpy uint8
                image_np = np.array(image, dtype=np.uint8)
                
                # Store
                data_list.append({
                    'image': image_np,
                    'label': label.numpy()
                })
            except Exception as e:
                print(f"Error processing index {i}: {e}")
                continue
            
        # Save pickle
        save_path = os.path.join(output_dir, f"{split_name}.pkl")
        print(f"Saving to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(data_list, f)
            
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
