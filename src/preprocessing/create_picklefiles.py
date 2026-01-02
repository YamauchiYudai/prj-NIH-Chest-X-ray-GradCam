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
    
    # Chunk size for saving multiple pickle files to avoid RAM crash
    CHUNK_SIZE = 2000

    for split_name, (files, labels) in splits.items():
        if len(files) == 0:
            print(f"Skipping {split_name} (empty)")
            continue
            
        print(f"Processing {split_name} split ({len(files)} images)...")
        
        # Instantiate dataset to reuse path finding logic
        # We pass None as transform so we get the PIL image
        dataset = NIHChestXrayDataset(files, labels, class_map, data_root, transform=None)
        
        total_images = len(dataset)
        num_chunks = (total_images + CHUNK_SIZE - 1) // CHUNK_SIZE

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = min((chunk_idx + 1) * CHUNK_SIZE, total_images)
            
            print(f"  Processing chunk {chunk_idx + 1}/{num_chunks} (indices {start_idx} to {end_idx})...")
            
            data_list = []
            for i in tqdm(range(start_idx, end_idx), leave=False):
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
                
            # Save pickle chunk
            save_path = os.path.join(output_dir, f"{split_name}_part_{chunk_idx}.pkl")
            # If there's only one chunk, maybe we want to keep the original name? 
            # But for consistency, let's use parts or handle single file. 
            # If we want to support the old way, we could check num_chunks. 
            # But let's stick to parts for consistency in the new loader.
            
            print(f"  Saving to {save_path}...")
            with open(save_path, 'wb') as f:
                pickle.dump(data_list, f)
            
            # Free memory
            del data_list
            
        print(f"Finished {split_name}")

if __name__ == "__main__":
    main()
