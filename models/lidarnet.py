import torch
import torch.nn as nn
import torch.nn.functional as F


class LidarNet(nn.Module):
    def __init__(self, input_shape, num_classes=256, channel=32, drop_prob=0.3, fusion=True):
        super(LidarNet, self).__init__()
        self.fusion = fusion
        self.drop_prob = drop_prob


        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)


        self.conv4 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)


        self.conv6 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2))


        self.conv8 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1)


        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 512)
        self.dropout4 = nn.Dropout(0.2)

        if fusion:
            self.fc2 = nn.Linear(512, 256)
            self.dropout5 = nn.Dropout(0.2)
            self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):

        a = F.relu(self.conv1(x))
        x = F.relu(self.conv2(a))
        x = F.relu(self.conv3(x))
        x = self.maxpool1(x + a)
        x = F.dropout(x, self.drop_prob)


        b = F.relu(self.conv4(x))
        x = F.relu(self.conv5(b))
        x = self.maxpool2(x + b)
        x = F.dropout(x, self.drop_prob)


        c = F.relu(self.conv6(x))
        x = F.relu(self.conv7(c))
        x = self.maxpool3(x + c)
        x = F.dropout(x, self.drop_prob)


        d = F.relu(self.conv8(x))
        x = F.relu(self.conv9(d))
        x = x + d


        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)

        if self.fusion:
            x = F.relu(self.fc2(x))
            x = self.dropout5(x)
            x = torch.tanh(self.fc_out(x))

        return x