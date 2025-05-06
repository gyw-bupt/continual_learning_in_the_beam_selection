import torch
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.x_coord_data = np.load(data_file)['coord']
        self.x_lidar_data = np.load(data_file)['lidar']
        self.y_data = np.load(data_file)['y']
        self.transform = transform

    def __len__(self):
        return len(self.x_coord_data)

    def __getitem__(self, idx):
        x_coord = self.x_coord_data[idx]
        x_lidar = self.x_lidar_data[idx]
        y = self.y_data[idx]


        x_coord = torch.tensor(x_coord, dtype=torch.float32)
        x_lidar = torch.tensor(x_lidar, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x_coord, x_lidar, y


