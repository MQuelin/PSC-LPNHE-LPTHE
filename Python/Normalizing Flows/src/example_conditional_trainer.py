import torch

from flows import ConditionalNF, CIAF
from datasets import TestSet2
from train import ConditionalTrainer
from tests import sample_flow_discreet


import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# flow = CIAF(4, 1, 3)
flow = torch.load('Python/Normalizing Flows/models/test.pt')
sample_flow_discreet(flow,100,show_train_set=False)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)
data = TestSet2(50000, plot_data=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

### Dividing the data on train and test
percentage_train = 0.8
train_len = int(percentage_train * len(data))
data_train, data_test = torch.utils.data.random_split(data, 
                                                      [train_len, len(data)-train_len])

nb_epochs = 100
batch_size = 10000

dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

trainer = ConditionalTrainer(flow, optimizer, dataloader, dataloader_test, 1, 3, 0.05, device)

loss_train, loss_test = trainer.train(nb_epochs)

trainer.save_at(save_path= "../models", save_name="test.pt")

plt.plot(range(len(loss_train)), torch.log(torch.Tensor(loss_train)))
plt.show()
plt.plot(range(len(loss_test)), torch.log(torch.Tensor(loss_test)))
plt.show()

# print(loss_train)
# print(loss_test)