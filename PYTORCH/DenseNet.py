import torch
from torch import nn


# Dense layer

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels,growth_rate,kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = torch.cat([x, out], dim=1)
        return out
    
    

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.dense_block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.dense_block(x)



#Transition Layer(for downsampling)
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        return self.transition(x)




class DenseNet(nn.Module):
    def __init__(self, gorwth_rate=32, num_classes=10):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #Dense Blocks with transition layers
        
        self.block1 = DenseBlock(6, 64, gorwth_rate)
        self.trans1 = TransitionLayer(64 + 6 * gorwth_rate, 128)
        
        self.block2 = DenseBlock(12, 128, gorwth_rate)
        self.trans2 = TransitionLayer(128 + 12 * gorwth_rate, 256)
        
        self.block3 = DenseBlock(24, 256, gorwth_rate)
        self.trans3 = TransitionLayer(256 + 24 * gorwth_rate, 512)
        
        self.block4 = DenseBlock(16, 512, gorwth_rate)
        self.bn_final = nn.BatchNorm2d(512 + 16*gorwth_rate)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 + 16*gorwth_rate, num_classes)
        

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.trans3(self.block3(x))
        x = self.block4(x)
        x = self.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


if __name__ == "__main__":
    model = DenseNet(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    
    print(f"Output Shape: {output.shape}")




















