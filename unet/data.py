from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
from pathlib import Path
from PIL import Image
import numpy as np
class P3MDataset(Dataset):
    def __init__(self, input_path:Path, label_path: Path) -> None:
        super().__init__()
        self.input_path = input_path
        self.label_path = label_path
        self.images_name = [image.stem for image in input_path.iterdir()]

    def __len__(self):
        return sum(1 for _ in self.input_path.iterdir())
    
    def __getitem__(self, idx:int):
        img_name = self.images_name[idx]
        input_path = self.input_path / f"{img_name}.jpg"
        label_path = self.label_path / f"{img_name}.png"
        image = np.array(Image.open(input_path))
        label = np.array(Image.open(label_path))
        return image, label

