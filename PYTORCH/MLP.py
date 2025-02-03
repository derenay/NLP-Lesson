import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# 1. Load the Fashion-MNIST Dataset
batch_size = 256

# Convert images to tensors
transform = transforms.ToTensor()

# Load datasets
train_data = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_data = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

# Create data loaders
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 2. Define the MLP Model
net = nn.Sequential(
    nn.Flatten(),  # Flatten 28x28 images into 1D (784)
    nn.Linear(784, 256),  # Hidden layer with 256 neurons
    nn.ReLU(),  # Activation function
    nn.Linear(256, 10)  # Output layer (10 classes)
)

# 3. Initialize Model Parameters
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 4. Define Loss Function (Cross-Entropy for Classification)
loss = nn.CrossEntropyLoss()

# 5. Define Optimizer (SGD)
lr = 0.1  # Learning rate
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 6. Training the Model
num_epochs = 10

for epoch in range(num_epochs):
    for X, y in train_loader:
        # Flatten the images
        X = X.view(-1, 28 * 28)  

        # Forward pass: Compute predictions
        y_hat = net(X)

        # Compute loss
        l = loss(y_hat, y)

        # Backpropagation
        trainer.zero_grad()
        l.backward()
        trainer.step()

    # Evaluate accuracy on training and test sets
    train_acc = sum((net(X.view(-1, 784)).argmax(dim=1) == y).sum() for X, y in train_loader) / len(train_data)
    test_acc = sum((net(X.view(-1, 784)).argmax(dim=1) == y).sum() for X, y in test_loader) / len(test_data)

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")
