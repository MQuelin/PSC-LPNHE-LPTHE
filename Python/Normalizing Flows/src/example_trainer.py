import torch

from flows import SimplePlanarNF
from data_loaders import ZeeDataset
from train import NormTrainer


import matplotlib.pyplot as plt

flow = SimplePlanarNF(4, 6)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
data = ZeeDataset('../data/data_dict.pkl')
device = 'cuda'

nb_epochs = 100
batch_size = 10000

trainer = NormTrainer(flow, optimizer, data.KDE_of_outputs, 6, device)

loss = trainer.train_and_save(nb_epochs, batch_size)

plt.plot(range(nb_epochs), loss)
plt.show()