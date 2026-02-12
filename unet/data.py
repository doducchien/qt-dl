from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
from pathlib import Path
from PIL import Image
import numpy as np

from torchvision.transforms import v2

class P3MDataset(Dataset):
    def __init__(self, input_path:Path, label_path: Path, transform: None) -> None:
        super().__init__()
        self.input_path = input_path
        self.label_path = label_path
        self.transform = transform
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.images_name = [image.stem for image in input_path.iterdir()]

    def __len__(self):
        return sum(1 for _ in self.input_path.iterdir())
    
    def __getitem__(self, idx:int):
        img_name = self.images_name[idx]
        input_path = self.input_path / f"{img_name}.jpg"
        label_path = self.label_path / f"{img_name}.png"

        image = Image.open(input_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        if self.transform:
            image, label = self.transform(image, label)
        image = self.normalize(image)
        return image, label

