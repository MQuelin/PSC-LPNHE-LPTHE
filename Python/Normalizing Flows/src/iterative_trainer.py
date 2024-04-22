import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from flows import *
from datasets import ZeeDataset
from train import Trainer


"""
This training algorithm functions in a similar way to the example trainer but it allows for variable
learning rates during training
"""

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float64)

data = ZeeDataset('../data/100k2.pkl')
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'selected device {device}')

flow = torch.load('Python/Normalizing Flows/models/IAF2_step_4.pt')
#flow = SimpleIAF2(10,10,device)
flow_name = "IAF2"

### Dividing the data on train and test
batch_size = 2000

percentage_train = 0.8
train_len = int(percentage_train * len(data))
data_train, data_test = torch.utils.data.random_split(data,[train_len, len(data)-train_len])

dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

nb_epochs = 2
lr_list = [1e-8,1e-8,1e-8,1e-8,1e-8]

train_losses = []
test_losses = []

k=0
for learning_rate in tqdm(lr_list, desc='Performing interative training'):

    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)

    trainer = Trainer(flow, optimizer, dataloader, dataloader_test, 10, device)

    train_loss, test_loss = trainer.train(nb_epochs)
    train_losses += train_loss
    test_losses += test_loss

    trainer.save_at(save_name=flow_name + f'_step_{k}.pt')
    k += 1


plt.plot(range(len(train_losses)), torch.log(torch.Tensor(train_losses)))
plt.show()
plt.plot(range(len(test_losses)), torch.log(torch.Tensor(test_losses)))
plt.show()