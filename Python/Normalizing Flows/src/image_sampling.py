from flows import ImageConditionalNF
import torchvision as tv
from torchvision.datasets import MNIST
import torch.nn as nn

import torch


import matplotlib.pyplot as plt

from pathlib import Path

flow = torch.load(Path(__file__) / Path('../../models/ICNF_9.pt'))

# definition of transform func that will convert images to vectors
transform = tv.transforms.ToTensor()

# Dataset path
PATH = Path(__file__) / Path("../../data/MNIST")

# Fetching the MNIST data from torchvision
train_set = MNIST(PATH,
                 train=True,
                 download=True,
                 transform=transform)

test_set = MNIST(PATH,
                 train=False,
                 download=True,
                 transform=transform)
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_set), size=(1,)).item()
    img, label = train_set[sample_idx]
    print(img)
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    X = torch.squeeze(flow.sample(torch.Tensor([[float(i)] for k in range(2)])))
    transform1 = nn.Threshold(threshold=-1,value=-1)
    transform2 = nn.Threshold(threshold=0,value=0)
    X = transform2(-transform1(-X))
    # print(X[0])
    figure.add_subplot(rows, cols, i)
    plt.title(str(i))
    plt.axis("off")
    plt.imshow(X.to('cpu').detach()[0], cmap="gray")
plt.show()