import torch

from flows import ConditionalNF
from datasets import ZeeDataset
from train import ConditionalTrainer


import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

flow = ConditionalNF(32, 3, 10, [10,10,10])
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)
data = ZeeDataset('../data/100k2.pkl')
device = "cuda" if torch.cuda.is_available() else "cpu"

### Dividing the data on train and test
percentage_train = 0.8
train_len = int(percentage_train * len(data))
data_train, data_test = torch.utils.data.random_split(data, 
                                                      [train_len, len(data)-train_len])

nb_epochs = 100 #last around -1.5
batch_size = 1000

dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

trainer = ConditionalTrainer(flow, optimizer, dataloader, dataloader_test, 3, 10, 0.05, device)

loss_train, loss_test = trainer.train(nb_epochs)

trainer.save_at(save_path= "../models", save_name="CNF2_test_3.pt")

plt.plot(range(len(loss_train)), torch.log(torch.Tensor(loss_train)))
plt.show()
plt.plot(range(len(loss_test)), torch.log(torch.Tensor(loss_test)))
plt.show()

#print(loss_train)