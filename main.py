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
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- 1. Load Data ---
    try:
        dataloaders, train_df = get_dataloaders(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[Error] Could not load data: {e}")
        print("\nPlease ensure the dataset is downloaded and the `data_dir` in `conf/dataset/vindr.yaml` is correct.")
        return # Exit gracefully

    # --- 2. Get Model ---
    print(f"\nLoading model: {cfg.model.name}")
    model = get_model(cfg).to(device)

    # --- 3. Define Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 4. Training ---
    print("\nStarting training...")
    best_model = train_model(
        model, 
        criterion, 
        optimizer, 
        dataloaders, 
        device, 
        num_epochs=cfg.epochs,
        scheduler=exp_lr_scheduler
    )

    # --- 5. Evaluation (on test set) ---
    print("\nRunning evaluation on the test set...")
    best_model.eval()
    test_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = best_model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
    
    test_acc = test_corrects.double() / len(dataloaders['test'].dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    # --- 6. Generate Grad-CAM Visualization ---
    try:
        # Generate a random grid from the test set
        generate_gradcam_grid(best_model, dataloaders['test'], cfg)
        # Generate specific CAMs for each pathology using the training set samples
        generate_gradcam_by_pathology(best_model, train_df, cfg, device)
    except Exception as e:
        print(f"\n[Error] Failed to generate Grad-CAM: {e}")

    print(f"\nExecution finished. Check the output directory: {os.getcwd()}")


if __name__ == "__main__":
    main()
