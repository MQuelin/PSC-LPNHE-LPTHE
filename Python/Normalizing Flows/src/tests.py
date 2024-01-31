import pickle
from pathlib import Path
import torch

absolute_path = Path(__file__).parent
relative_path = '../data/data_dict.pkl'
open_path = absolute_path / relative_path

with open(open_path, 'rb') as f:
    data_dict = pickle.load(f)
    print('Now loading dataset ...')

print('Done')

a = torch.tensor([[1.,-1.,3.],[4.,50.,6.]])
a_mean = torch.mean(a, dim=0,)
a_std = torch.std(a, dim=0)

print((a-a_mean)/a_std)