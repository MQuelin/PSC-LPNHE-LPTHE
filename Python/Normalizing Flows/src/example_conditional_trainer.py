import torch

from flows import ConditionalNF
from datasets import ZeeDataset
from train import ConditionalTrainer


import matplotlib.pyplot as plt

flow = ConditionalNF(24, 3, 6)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
data = ZeeDataset('../data/data_dict.pkl')
device = 'cuda'

nb_epochs = 10
batch_size = 2000

dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

trainer = ConditionalTrainer(flow, optimizer, dataloader, 3, 6, 1e-2, device)

loss = trainer.train(nb_epochs)

trainer.save_at(save_name="Conditionnal_Test_1")

plt.plot(range(len(loss)), loss)
plt.show()