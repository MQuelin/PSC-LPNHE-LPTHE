from datasets import ZeeDataset
import matplotlib.pyplot as plt
import torch
from Sacha.src.glow_pytorch.My_model import Glow

data = ZeeDataset('../data/100k2.pkl')
device = "cuda" if torch.cuda.is_available() else "cpu"

### Dividing the data on train and test
percentage_train = 0.8
train_len = int(percentage_train * len(data))
data_train, data_test = torch.utils.data.random_split(data, 
                                                      [train_len, len(data)-train_len])

nb_epochs = 10
batch_size = 1000

dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

trainer = Glow(in_channel=3, n_flow=1, n_block=5, affine=True, conv_lu=True)

trainer.train()

trainer.save_at(save_path= "../models", save_name="Glow_test_3_1_T_T.pt")


#plt.plot(range(len(loss_train)), loss_train)
#plt.show()
#plt.plot(range(len(loss_test)), loss_test)
#plt.show()