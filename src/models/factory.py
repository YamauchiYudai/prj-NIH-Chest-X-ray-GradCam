import torch
import torch.nn as nn
from torchvision import models
import timm
from omegaconf import DictConfig
from typing import Optional

def get_model(cfg: DictConfig) -> nn.Module:
    """
    Model factory for creating a model based on the configuration.

    Args:
        cfg (DictConfig): A DictConfig object with model and dataset settings.
                          Expects `cfg.model.name`, `cfg.model.pretrained`,
                          and `cfg.dataset.num_classes`.

    Returns:
        nn.Module: The constructed model.
        
    Raises:
        ValueError: If the model name is not supported.
    """
    model_name = cfg.model.name
    num_classes = cfg.dataset.num_classes
    pretrained = cfg.model.pretrained
    
    model: Optional[nn.Module] = None

    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'efficientnet_b0':
        # Using timm for EfficientNet as it's very flexible
        model = timm.create_model(
            'efficientnet_b0', 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        # timm automatically handles the classifier replacement
        
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    return model

if __name__ == '__main__':
    # A simple test to check if the factory works
    from omegaconf import OmegaConf

    # Mock config for ResNet50
    cfg_resnet = OmegaConf.create({
        'model': {'name': 'resnet50', 'pretrained': True},
        'dataset': {'num_classes': 3}
    })
    model_resnet = get_model(cfg_resnet)
    print(f"Successfully created resnet50: {model_resnet.__class__.__name__}")
    print(model_resnet.fc)
    
    # Mock config for DenseNet121
    cfg_densenet = OmegaConf.create({
        'model': {'name': 'densenet121', 'pretrained': True},
        'dataset': {'num_classes': 3}
    })
    model_densenet = get_model(cfg_densenet)
    print(f"\nSuccessfully created densenet121: {model_densenet.__class__.__name__}")
    print(model_densenet.classifier)

    # Mock config for EfficientNet-B0
    cfg_effnet = OmegaConf.create({
        'model': {'name': 'efficientnet_b0', 'pretrained': True},
        'dataset': {'num_classes': 3}
    })
    model_effnet = get_model(cfg_effnet)
    print(f"\nSuccessfully created efficientnet_b0: {model_effnet.__class__.__name__}")
    print(model_effnet.get_classifier())

    # Test with a dummy input
    dummy_input = torch.randn(4, 3, 224, 224) # (batch, channels, height, width)
    output = model_resnet(dummy_input)
    print(f"\nResNet50 output shape: {output.shape}") # Expected: [4, 3]

    output = model_densenet(dummy_input)
    print(f"DenseNet121 output shape: {output.shape}") # Expected: [4, 3]

    output = model_effnet(dummy_input)
    print(f"EfficientNet-B0 output shape: {output.shape}") # Expected: [4, 3]
