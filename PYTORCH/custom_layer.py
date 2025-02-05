import torch 
from torch import nn


class CustomLayer(nn.Module):
    def __init__(self):
        super(CustomLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(1))
        
        
    def forward(self, x):
        return x * self.weight
    
    
class CustomModel(nn.Module):
    def __init__(self, input_dim):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 10)
        self.custom_layer = CustomLayer()
        
    def forward(self, x):
        x = self.layer1(x)
        return self.custom_layer(x)
        
model = CustomModel(input_dim=5)
print(model)


























