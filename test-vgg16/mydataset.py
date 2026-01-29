import os.path
import torchvision.transforms as transforms

import pandas as pd
from torch.utils.data import  Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, csv_file_dir: str, images_dir: str, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.csv_data = pd.read_csv(csv_file_dir)
        self.transform = transform

        labels = sorted(self.csv_data['label'].unique())
        self.labels_idx ={label: i for i, label in enumerate(labels)}
        print(self.labels_idx)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_name = f"{self.csv_data.iloc[idx,0]}.png"
        img_dir = os.path.join(self.images_dir, img_name)
        label = self.csv_data.iloc[idx,1]
        idx_label = self.labels_idx[label]

        image = Image.open(img_dir)
        if self.transform:
            image = self.transform(image)

        return image, idx_label
