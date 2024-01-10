import pickle
import numpy
import torch
from sklearn.neighbors import KernelDensity
from scipy.stats import uniform
from pathlib import Path

data = torch.tensor(uniform.rvs(size=(10000,2)))

print(data.shape)