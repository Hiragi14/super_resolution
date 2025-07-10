import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(base_channels, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, in_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)
