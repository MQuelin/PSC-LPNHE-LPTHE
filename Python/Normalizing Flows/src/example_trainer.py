import torch
import matplotlib.pyplot as plt

from flows import *
from datasets import ZeeDataset
from train import Trainer


torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

data = ZeeDataset('../data/100k2.pkl')
device = "cuda" if torch.cuda.is_available() else "cpu"

flow = torch.load('Python/Normalizing Flows/models/IAF_iter.pt')

optimizer = torch.optim.Adam(flow.parameters(), lr=1e-5)

### Dividing the data on train and test
percentage_train = 0.8
train_len = int(percentage_train * len(data))
data_train, data_test = torch.utils.data.random_split(data,[train_len, len(data)-train_len])

nb_epochs = 10
batch_size = 2000

dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

trainer = Trainer(flow, optimizer, dataloader, dataloader_test, 10, device)

train_loss, test_loss = trainer.train(nb_epochs)

trainer.save_at(save_name="IAF_iter.pt")

plt.plot(range(len(train_loss)), torch.log(torch.Tensor(train_loss)))
plt.show()
plt.plot(range(len(test_loss)), torch.log(torch.Tensor(test_loss)))
plt.show()