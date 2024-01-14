import torch

from flows import SimplePlanarNF
from data_loaders import ZeeDataset, TempUniform
from train import NormTrainer


import matplotlib.pyplot as plt

flow = SimplePlanarNF(64, 2)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)
data = TempUniform(data_dim=2)
device = 'cpu'

nb_epochs = 300
batch_size = 10000

trainer = NormTrainer(flow, optimizer, data.KDE_of_outputs, 2, device)

loss = trainer.train_and_save(nb_epochs, batch_size)

plt.plot(range(nb_epochs), loss)
plt.show()