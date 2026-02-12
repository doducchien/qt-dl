import torch
import numpy as np
from torch.utils.data import DataLoader
from UNET import UNET
from torch.utils.tensorboard import SummaryWriter
import torchvision

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def un_normalize(batch_inputs: torch.Tensor):
    batch_mean = torch.from_numpy(mean).reshape(1,-1,1,1)
    batch_std = torch.from_numpy(std).reshape(1,-1,1,1)

    batch_inputs_copy = batch_inputs.clone()
    batch_inputs_copy.mul_(batch_std).add_(batch_mean)
    return torch.clamp(batch_inputs_copy)

def get_random_test_images(data_loader: DataLoader, size:int):
    indices = list(range(size))
    np.random.shuffle(indices)
    images_labels_test = torch.tensor()
    for idx in indices:
        images_labels_test.append(data_loader.dataset[idx])
    images_labels_test = torch.tensor(images_labels_test)
    return images_labels_test


def visualize(
        model: UNET, 
        data_loader:DataLoader, 
        device:str, 
        num_images:int, 
        writer: SummaryWriter,
        step:int
    ):
    model.eval()
    images, masks = next(iter(data_loader))

    images = images[:num_images].to(device)
    masks = masks[:num_images].to(device)

    with torch.no_grad():
        preds = model(images)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
    
    mask_rgb = masks.repeat(1,3,1,1)
    preds_rgb = preds.repeat(1,3,1,1)

    combined = torch.stack([images, mask_rgb, preds_rgb], dim=1)
    combined_tensor = combined.flatten(0, 1)
    grid_image = torchvision.utils.make_grid(combined_tensor, nrow=3, padding=2, normalize=False)
    writer.add_image("Visualize input/predict", grid_image, step)
