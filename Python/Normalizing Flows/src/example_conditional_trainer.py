import torch

from flows import ConditionalNF
from datasets import ZeeDataset
from train import ConditionalTrainer


import matplotlib.pyplot as plt

flow = ConditionalNF(50, 3, 10)
optimizer = torch.optim.Adam(flow.parameters(), lr=5e-4)
data = ZeeDataset('../data/data_dict.pkl')
device = 'cuda'

nb_epochs = 30
batch_size = 1000

dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

trainer = ConditionalTrainer(flow, optimizer, dataloader, 3, 10, 1e-2, device)

loss = trainer.train(nb_epochs)

trainer.save_at(save_path= "../models", save_name="ConditionalNF_50layers_10kZee_noRings_310124.pt")

plt.plot(range(len(loss)), loss)
plt.show()

print(loss)