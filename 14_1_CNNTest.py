import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 4

batch_size = 4
learning_rate = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

train_dataset = torchvision.datasets.CIFAR10(root='./data', train = True, download= True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train = False, download= True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle= True) #또 좀 수상
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle= False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # CHW -> HWC
    plt.show()


dataiter = iter(train_loader)
images, labels = next(dataiter) 

imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

conv1 = nn.Conv2d(3,6,5)
pool = nn.MaxPool2d(2,2)
conv2 = nn.Conv2d(6,16,5)
print(images.shape)

x= conv1(images)
x= pool(x)
x=conv2(x)
x=pool(x)

