import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionNet(nn.Module):
    def __init__(self, lidar_model, coord_model, num_classes=256):
        super(FusionNet, self).__init__()
        self.lidar_model = lidar_model
        self.coord_model = coord_model

        self.linear = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=30, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=30, out_channels=30, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(30),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=30, out_channels=30, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=30, out_channels=30, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )


        self.last = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1680, num_classes * 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(num_classes * 3, num_classes * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(num_classes * 2, num_classes),
        )

    def features(self, lidar_input, coord_input):
        lidar_output = self.lidar_model(lidar_input)
        coord_output = self.coord_model(coord_input)
        combined_output = torch.cat((lidar_output, coord_output), dim=1)
        z = combined_output.view(-1, 2, 256)
        z = self.linear(z)
        return z

    def logits1(self, z):
        out = self.last(z)
        return out

    def forward(self, lidar_input, coord_input):
        z = self.features(lidar_input, coord_input)
        out = self.logits1(z)
        return out