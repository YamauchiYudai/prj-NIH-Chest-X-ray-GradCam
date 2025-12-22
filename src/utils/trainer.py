import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import copy
from typing import Dict
import os

def train_model(
    model: nn.Module,
    criterion,
    optimizer,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    num_epochs: int = 10,
    scheduler=None
):
    """
    Handles the main training and validation loop with TensorBoard logging.

    Args:
        model (nn.Module): The model to be trained.
        criterion: The loss function.
        optimizer: The optimization algorithm.
        dataloaders (Dict[str, DataLoader]): Dictionary with 'train' and 'val' DataLoaders.
        device (torch.device): The device to run training on (CPU or CUDA).
        num_epochs (int): Total number of epochs to train for.
        scheduler: Learning rate scheduler.

    Returns:
        nn.Module: The best performing model weights.
    """
    since = time.time()
    
    # Initialize TensorBoard writer
    # It will log to a directory within the current working directory (Hydra's output dir)
    writer = SummaryWriter(log_dir='.')

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log metrics to TensorBoard
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model checkpoint
                torch.save(model.state_dict(), 'best_model.pth')
                print("Saved best model checkpoint to best_model.pth")
        
        # Log learning rate
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.flush()
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    writer.close()
    return model
