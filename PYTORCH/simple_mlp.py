import torch 
from torch import nn


class SimpleMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLPModel, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim),
        self.output = nn.Linear(hidden_dim, output_dim),
        self.activation = nn.ReLU()
        
    def forward(self):
        x = self.activation(self.hidden)
        return self.output(x)


if __name__ == "__main__":
    model = SimpleMLPModel(input_dim=10, hidden_dim=20, output_dim=1)
    print(model)