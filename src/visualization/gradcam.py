import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import pandas as pd
from PIL import Image


def _get_target_layer_by_name(model: nn.Module, layer_name: str) -> Optional[nn.Module]:
    """Recursively finds a layer in a model by its string name."""
    modules = layer_name.split('.')
    target_layer = model
    for module_name in modules:
        if not hasattr(target_layer, module_name):
            return None
        target_layer = getattr(target_layer, module_name)
    return target_layer

def _visualize_and_save_cam(rgb_img: np.ndarray, cam_grayscale: np.ndarray, original_label: str, filename: str):
    """Helper to create and save a single CAM image plot."""
    cam_image = show_cam_on_image(rgb_img, cam_grayscale, use_rgb=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Original (Label: {original_label})")
    axes[0].axis('off')

    axes[1].imshow(cam_image)
    axes[1].set_title("Grad-CAM")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved Grad-CAM to {filename}")

def generate_gradcam_by_pathology(
    model: nn.Module,
    full_df: pd.DataFrame,
    cfg: DictConfig,
    device: torch.device
):
    """
    Generates a Grad-CAM visualization for one sample of each target pathology.

    Args:
        model (nn.Module): The trained model.
        full_df (pd.DataFrame): The complete dataframe of the dataset.
        cfg (DictConfig): Hydra configuration.
        device (torch.device): The device to run the model on.
    """
    print("\n--- Generating Grad-CAM for each pathology ---")
    model.eval()
    target_layer_name = cfg.model.target_layer
    target_layer = _get_target_layer_by_name(model, target_layer_name)
    if target_layer is None:
        print(f"Warning: Target layer '{target_layer_name}' not found. Skipping.")
        return
        
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Define transforms for a single image
    transform = transforms.Compose([
        transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class_map = {name: i for i, name in enumerate(cfg.dataset.classes)}

    for class_name, class_idx in class_map.items():
        # Find a sample image that contains the current pathology
        sample_row = full_df[full_df['Finding Labels'].str.contains(class_name, na=False)]
        if sample_row.empty:
            print(f"No sample found for pathology: {class_name}. Skipping.")
            continue
        
        sample = sample_row.iloc[0]
        image_path = sample['image_path']
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {class_name}, could not load image {image_path}: {e}")
            continue

        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Denormalize for visualization
        rgb_img_np = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_img_np = std * rgb_img_np + mean
        rgb_img_np = np.clip(rgb_img_np, 0, 1)

        # Generate CAM
        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Save visualization
        filename = f"gradcam_{class_name.replace(' ', '_')}.png"
        _visualize_and_save_cam(rgb_img_np, grayscale_cam, class_name, filename)


def generate_gradcam_grid(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: DictConfig,
    num_images: int = 4
):
    """
    Generates and saves a grid of Grad-CAM visualizations from a dataloader batch.
    """
    print("\n--- Generating random Grad-CAM grid ---")
    model.eval()
    target_layer__name = cfg.model.target_layer
    target_layer = _get_target_layer_by_name(model, target_layer_name)
    if target_layer is None:
        print(f"Warning: Target layer '{target_layer_name}' not found. Skipping.")
        return
        
    cam = GradCAM(model=model, target_layers=[target_layer])

    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    targets = [ClassifierOutputTarget(label) for label in labels]
    grayscale_cam = cam(input_tensor=images, targets=targets)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Create a single grid image
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 4, 8))
    
    for i in range(num_images):
        rgb_img = images[i].permute(1, 2, 0).cpu().numpy()
        rgb_img = std * rgb_img + mean # Denormalize
        rgb_img = np.clip(rgb_img, 0, 1)

        cam_image = show_cam_on_image(rgb_img, grayscale_cam[i, :], use_rgb=True)

        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f"Original (GT: {labels[i].item()})")
        axes[0, i].axis('off')

        axes[1, i].imshow(cam_image)
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis('off')

    plt.tight_layout()
    output_path = f"gradcam_grid_{cfg.model.name}.png"
    plt.savefig(output_path)
    print(f"Grad-CAM grid saved to: {output_path}")
    plt.close()
