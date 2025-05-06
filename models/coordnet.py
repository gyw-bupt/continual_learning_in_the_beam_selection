import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordNet(nn.Module):
    def __init__(self, input_shape, num_classes=256, fusion=True):
        super(CoordNet, self).__init__()
        self.fusion = fusion

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=20, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=10, kernel_size=2, padding=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, padding=1)

        self.conv3 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=20, out_channels=10, kernel_size=2, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(30, 1024)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.25)

        self.output = nn.Linear(512, num_classes)

    def forward(self, x):

        x = x.transpose(1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool2(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        if self.fusion:
            x = torch.tanh(self.output(x))

        return x