import torch

from flows import SimpleAdditiveNF
from datasets import ZeeDataset
from train import Trainer


import matplotlib.pyplot as plt

flow = SimpleAdditiveNF(24, 6)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
data = ZeeDataset('../data/data_dict.pkl')
device = 'cuda'

nb_epochs = 15
batch_size = 2000

dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

trainer = Trainer(flow, optimizer, dataloader, 6, device)

loss = trainer.train(nb_epochs)

trainer.save_at(save_name="Test_1")

plt.plot(range(len(loss)), loss)
plt.show()