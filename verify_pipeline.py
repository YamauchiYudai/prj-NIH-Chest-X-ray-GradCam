import torch
import torch.nn as nn
from omegaconf import OmegaConf
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Imports from our project
from src.data.dataset import get_dataloaders
from src.models.factory import get_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def get_target_layer(model, model_name):
    if model_name == 'resnet50':
        return model.layer4[-1]
    elif model_name == 'densenet121':
        return model.features[-1]
    elif model_name == 'efficientnet_b0':
        return model.conv_head
    else:
        return None

def verify():
    print("Starting Pipeline Verification...")
    
    # 1. Setup Configuration (Overriding with mock values for verification)
    cfg = OmegaConf.create({
        'dataset': {
            'data_dir': './data',
            'classes': ['Atelectasis', 'Cardiomegaly', 'Effusion'],
            'num_classes': 3,
            'image_size': 224,
            'batch_size': 2,
            'num_workers': 0
        },
        'model': {
            'name': 'resnet50',
            'pretrained': True,
            'num_classes': 3,
            'target_layer': 'layer4.2' # ResNet50 last bottleneck
        },
        'seed': 42
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data: Create DataLoader with debug=True (loads only 2 samples)
    print("Loading data (debug=True)...")
    dataloaders, metadata = get_dataloaders(cfg, debug=True)
    
    if 'train' not in dataloaders:
        print("Error: No data loaded. Check if data/train_val_list.txt exists.")
        # Create a dummy batch if no data exists for testing the rest of the flow
        print("Creating dummy batch for testing...")
        images = torch.randn(2, 3, 224, 224)
        labels = torch.zeros((2, 3))
    else:
        train_loader = dataloaders['train']
        images, labels = next(iter(train_loader))
    
    print(f"Batch loaded. Image shape: {images.shape}, Label shape: {labels.shape}")

    # 3. Model: Load ResNet50
    print(f"Loading model: {cfg.model.name}...")
    model = get_model(cfg).to(device)
    model.eval()

    # 4. Inference: Run one batch
    print("Running inference...")
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
    print(f"Inference successful. Output shape: {outputs.shape}")

    # 5. Grad-CAM: Execute on the first image of the batch
    print("Generating Grad-CAM...")
    # Get target layer for Grad-CAM
    target_layer = get_target_layer(model, cfg.model.name)
    if target_layer is None:
        print("Target layer not found for model.")
        return

    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # We take the first image and the class with highest probability or first class
    input_tensor = images[0].unsqueeze(0)
    target_category = outputs[0].argmax().item()
    targets = [ClassifierOutputTarget(target_category)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # Denormalize image for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_img = images[0].permute(1, 2, 0).cpu().numpy()
    rgb_img = std * rgb_img + mean
    rgb_img = np.clip(rgb_img, 0, 1)

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 6. Output: Save result
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(cam_image)
    axes[1].set_title(f"Grad-CAM (Class: {cfg.dataset.classes[target_category]})")
    axes[1].axis('off')
    
    output_fn = "verification_result.png"
    plt.tight_layout()
    plt.savefig(output_fn)
    plt.close()
    
    print(f"Verification Completed: Saved to {output_fn}")

if __name__ == "__main__":
    verify()
