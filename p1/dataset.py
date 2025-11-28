"""
PyTorch Dataset for orientations. Optionally returns crops (for training aggregation-aware model).
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import Tuple

class OrientationDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform or T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row.filename)
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        label = [0,90,180,270].index(int(row.angle))
        return x, label