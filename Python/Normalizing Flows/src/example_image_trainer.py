from flows import ImageConditionalNF
from train import MNISTImageTrainer

import torch
from torch.utils.data import DataLoader

import torchvision as tv
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

from pathlib import Path

torch.set_default_dtype(torch.float64)

# Device selection
device = torch.device('cuda')

# Random seed for reproductibility
seed = 2
torch.manual_seed(seed)

# Training epochs
n_epochs = 25

# Training batch size
batch_size = 1024

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

# Creating a map of label names
labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_set), size=(1,)).item()
    img, label = train_set[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

## Initializing loaders

train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(test_set,
                          batch_size=batch_size,
                          shuffle=False)

flow = ImageConditionalNF(24, 1, 28, MLP_shape_list=[40,40])
optimizer = torch.optim.Adam(flow.parameters(), lr=5e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"

trainer = MNISTImageTrainer(flow, optimizer, train_loader, test_loader, 1, 28*28, 0.01, device)

loss_train, loss_test = trainer.train(n_epochs)

trainer.save_at(save_path= "../models", save_name="ICNF_9.pt")

plt.plot(range(len(loss_train)), torch.log(torch.Tensor(loss_train)))
plt.show()
plt.plot(range(len(loss_test)), torch.log(torch.Tensor(loss_test)))
plt.show()

#print(loss_train)