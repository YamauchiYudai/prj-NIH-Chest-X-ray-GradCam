import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
from PIL import Image
from torchvision import transforms
import os

from src.models.factory import get_model

def predict(cfg: DictConfig, model_path: str, image_path: str):
    """
    Runs inference on a single image using a trained model.

    Args:
        cfg (DictConfig): Hydra config to instantiate the model.
        model_path (str): Path to the trained model checkpoint (.pth).
        image_path (str): Path to the input image.
    """
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Build Model ---
    # We need to know the model architecture, so we load the config from the training run
    print(f"Loading model architecture: {cfg.model.name}")
    model = get_model(cfg).to(device)

    # --- 3. Load Trained Weights ---
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {model_path}")
        return
    model.eval()

    # --- 4. Load and Preprocess Image ---
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        return

    # Use the same transforms as validation
    preprocess = transforms.Compose([
        transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device) # Create a mini-batch as expected by the model

    # --- 5. Run Prediction ---
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    # --- 6. Display Results ---
    class_names = cfg.dataset.classes
    predicted_class = class_names[predicted_idx.item()]

    print(f"\n--- Prediction Results for {os.path.basename(image_path)} ---")
    print(f"Predicted Pathology: {predicted_class}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")
    print("\nConfidence scores per class:")
    for i, class_name in enumerate(class_names):
        print(f"  - {class_name}: {probabilities[i].item() * 100:.2f}%")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Hydra wrapper to get the config for prediction.
    The actual logic is in the predict function, which is called from the command line parser.
    """
    # This main is just for hydra to provide the config.
    # The parser below will override the flow.
    print("Hydra config loaded. Deferring to command-line argument parser.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument('model_path', type=str, help='Path to the trained model (.pth file).')
    parser.add_argument('image_path', type=str, help='Path to the input image file.')
    parser.add_argument(
        '--config_path', 
        type=str, 
        default='.', 
        help="Path to the Hydra output directory of the training run, which contains the `.hydra` config."
    )
    
    args = parser.parse_args()

    # We need to load the *exact* config used for training the model.
    # Hydra conveniently saves it in the output directory.
    try:
        # We manually compose the config from the directory where the model was saved
        with hydra.initialize_config_dir(config_dir=os.path.join(args.config_path, '.hydra')):
            cfg = hydra.compose(config_name="config")
    except Exception as e:
        print(f"Error loading Hydra config from '{args.config_path}': {e}")
        print("Please provide the path to a valid Hydra output directory (e.g., outputs/2023-12-22/10-00-00/)")
        exit()

    predict(cfg, args.model_path, args.image_path)
