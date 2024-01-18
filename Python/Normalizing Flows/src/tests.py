import torch
from sklearn.neighbors import KernelDensity
from scipy.stats import uniform
from pathlib import Path

data = torch.tensor([[1,2,3,4,5,6],[1,2,3,4,5,6]])
permutation_tensor = torch.LongTensor([0,1,2,3,5,4])

print(data)
print(data[:, permutation_tensor])