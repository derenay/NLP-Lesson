import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)

class ModernDenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ModernDenseNet, self).__init__()
        self.init_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = DenseBlock(64, growth_rate=32, num_layers=6)
        self.trans1 = TransitionLayer(64 + 32 * 6, 128)

        self.block2 = DenseBlock(128, growth_rate=32, num_layers=12)
        self.trans2 = TransitionLayer(128 + 32 * 12, 256)

        self.block3 = DenseBlock(256, growth_rate=32, num_layers=24)
        self.trans3 = TransitionLayer(256 + 32 * 24, 512)

        self.block4 = DenseBlock(512, growth_rate=32, num_layers=16)

        self.bn = nn.BatchNorm2d(512 + 32 * 16)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 + 32 * 16, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.pool(x)

        x = self.block1(x)
        x = self.trans1(x)

        x = self.block2(x)
        x = self.trans2(x)

        x = self.block3(x)
        x = self.trans3(x)

        x = self.block4(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x