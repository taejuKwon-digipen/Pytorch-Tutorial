# MNLIST
# DataLoader, Tranformation
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# hyper parameters
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

#MNIST
train_dataset = torchvision.datasets.MNIST(root= './data', train= True, transform=transforms.ToTensor(), download= True)
test_dataset = torchvision.datasets.MNIST(root= './data', train= False, transform= transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size=  batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size=  batch_size, shuffle = False)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)
