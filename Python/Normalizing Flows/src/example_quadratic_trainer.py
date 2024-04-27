import torch

from my_flows import ConditionalQS
from datasets import ZeeDataset
from train import ConditionalTrainer
from tests import is_cond_flow_invertible

import matplotlib.pyplot as plt

batch_size = 1000
#device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"

flow = ConditionalQS(2, 3, 10, 100, device=device)
optimizer = torch.optim.Adam(flow.parameters(), lr=5e-4)
data = ZeeDataset('../data/data_dict.pkl')

### Dividing the data on train and test
percentage_train = 0.8
train_len = int(percentage_train * len(data))
data_train, data_test = torch.utils.data.random_split(data, 
                                                      [train_len, len(data)-train_len])

nb_epochs = 5

dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

print(is_cond_flow_invertible(flow,verbose=True))

# trainer = ConditionalTrainer(flow, optimizer, dataloader, dataloader_test, 3, 10, 1e-2, device)

# loss_train, loss_test = trainer.train(nb_epochs)

# trainer.save_at(save_path= "../models", save_name="ConditionnalNF_QuadraticSpline_test.pt")

# print(is_cond_flow_invertible(flow))

# plt.plot(range(len(loss_train)), loss_train)
# plt.show()
# plt.plot(range(len(loss_test)), loss_test)
# plt.show()

#print(loss_train)