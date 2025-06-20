import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device ('cuda' if torch.cuda.is_available else 'cpu')

# Hyper-parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of nomalized range[-1, 1]

transform = transform.Compose(
    [transform.ToTensor(), 
     transform.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))]
)

train_dataset = torchvision.datasets.CIFAR10 (root= './data', train= True,
                                              download=True, transform= transform)
test_dataset = torchvision.datasets.CIFAR10 (root= './data', train= False,
                                              download=True, transform= transform)
train_loader = torch.utils.data.dataloader (train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.dataloader (test_dataset, batch_size = batch_size, shuffle = False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# implement conv net
class ConvNet(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
