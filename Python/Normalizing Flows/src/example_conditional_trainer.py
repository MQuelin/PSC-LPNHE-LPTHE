import torch

from flows import ConditionalNF
from datasets import ZeeDataset
from train import ConditionalTrainer


import matplotlib.pyplot as plt



flow = ConditionalNF(12, 3, 10)
optimizer = torch.optim.Adam(flow.parameters(), lr=5e-4)
data = ZeeDataset('../data/data_dict.pkl')
device = "cuda" if torch.cuda.is_available() else "cpu"

### Dividing the data on train and test
train_len = int(0.8 * len(data))
data_train, data_test = torch.utils.data.random_split(data, 
                                                      [train_len, len(data)-train_len])

nb_epochs = 30
batch_size = 1000

dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

trainer = ConditionalTrainer(flow, optimizer, dataloader, dataloader_test, 3, 10, 1e-2, device)

loss_train, loss_test = trainer.train(nb_epochs)

trainer.save_at(save_path= "../models", save_name="ConditionalNF_12layers_10kZee_noRings_310124.pt")

plt.plot(range(len(loss_train)), loss_train)
plt.show()
plt.plot(range(len(loss_test)), loss_test)
plt.show()

#print(loss_train)