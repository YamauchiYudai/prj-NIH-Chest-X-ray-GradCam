import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import random
import numpy as np

from src.data.dataset import get_dataloaders
from src.models.factory import get_model
from src.utils.trainer import train_model
from src.visualization.gradcam import generate_gradcam_by_pathology, generate_gradcam_grid

# Register OmegaConf resolvers
OmegaConf.register_new_resolver("len", len)
OmegaConf.register_new_resolver("to_int", int)

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # The two lines below are known to cause issues in some PyTorch versions/setups
    # especially with CUDA. Disable if you see performance issues.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def resolve_device(device_cfg: str) -> torch.device:
    """Resolves the device configuration to a torch.device object."""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training, evaluation, and visualization.
    """
    # Print the configuration - useful for debugging
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")
    
    # Set random seed
    set_seed(cfg.seed)
    
    # Setup devices
    train_device = resolve_device(cfg.train_device)
    test_device = resolve_device(cfg.test_device)
    
    print(f"Training device: {train_device}")
    print(f"Test/Inference device: {test_device}\n")

    # --- 1. Load Data ---
    try:
        dataloaders, train_df = get_dataloaders(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[Error] Could not load data: {e}")
        print("\nPlease ensure the dataset is downloaded and the `data_dir` in `conf/dataset/nih_chest_x_ray.yaml` is correct.")
        return # Exit gracefully

    # --- 2. Get Model (for Training) ---
    print(f"\nLoading model for training: {cfg.model.name}")
    model = get_model(cfg).to(train_device)

    # --- 3. Define Loss and Optimizer ---
    # Using BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 4. Training ---
    print("\nStarting training...")
    # train_model returns the model with best weights loaded (on train_device)
    # It also saves 'best_model.pth'
    train_model(
        model, 
        criterion, 
        optimizer, 
        dataloaders, 
        train_device, 
        num_epochs=cfg.epochs,
        scheduler=exp_lr_scheduler
    )

    # --- 5. Evaluation (on test set) ---
    print(f"\nRunning evaluation on the test set (Device: {test_device})...")
    
    # Re-initialize model for testing to ensure clean slate and correct device
    best_model = get_model(cfg).to(test_device)
    
    # Load best weights, handling device conversion (e.g. GPU -> CPU)
    # map_location ensures the weights are remapped to the target device
    print("Loading best model weights from 'best_model.pth'...")
    try:
        state_dict = torch.load('best_model.pth', map_location=test_device)
        best_model.load_state_dict(state_dict)
    except FileNotFoundError:
        print("Error: 'best_model.pth' not found. Training might have failed or not improved.")
        return

    best_model.eval()
    test_correct_bits = 0
    total_bits = 0
    
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(test_device)
            labels = labels.to(test_device)
            outputs = best_model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            test_correct_bits += torch.sum(preds == labels.data)
            total_bits += labels.numel()
    
    if total_bits > 0:
        test_acc = test_correct_bits.double() / total_bits
        print(f"Test Binary Accuracy: {test_acc:.4f}")
    else:
        print("Test set is empty.")

    # --- 6. Generate Grad-CAM Visualization ---
    try:
        # Generate a random grid from the test set
        # Note: generate_gradcam_grid likely needs to handle device moving internally 
        # or we pass the model already on the correct device.
        generate_gradcam_grid(best_model, dataloaders['test'], cfg)
        # Generate specific CAMs for each pathology using the training set samples
        generate_gradcam_by_pathology(best_model, train_df, cfg, test_device)
    except Exception as e:
        print(f"\n[Error] Failed to generate Grad-CAM: {e}")

    print(f"\nExecution finished. Check the output directory: {os.getcwd()}")


if __name__ == "__main__":
    main()
