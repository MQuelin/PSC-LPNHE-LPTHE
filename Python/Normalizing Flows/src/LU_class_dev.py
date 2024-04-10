import torch
from torch import nn
from numpy.random import permutation
import numpy as np

class LULayer(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        """
        w = torch.from_numpy(w)
        p, l, u = torch.linalg.lu(w)
        """

        self.du = nn.Parameter(torch.zeros(dim))
        self.D = torch.diag(torch.exp(self.du))

        self.u = nn.Parameter(torch.diag(torch.ones(dim)) + torch.triu(torch.randn(dim,  dim),diagonal=1))
        hu = self.u.register_hook(lambda grad: grad*torch.triu(torch.ones(dim,dim),diagonal=1))

        self.l = nn.Parameter(torch.diag(torch.ones(dim)) + torch.tril(torch.randn(dim, dim),diagonal=-1))
        hl = self.l.register_hook(lambda grad: grad*torch.tril(torch.ones(dim,dim),diagonal=-1))

        permutation_tensor = torch.LongTensor(permutation([i for i in range(dim)]))
        p_matrix = torch.zeros(dim, dim)
        for i, p in enumerate(permutation_tensor):
            p_matrix[i, p] = 1
        self.p = p_matrix

    
  
    def forward(self, x, reverse=False):
        w = self.p@self.l@self.u@self.D
        if not reverse : 
            y = (w @ x.T).T
            return y, torch.sum(self.du)
        else :
            y = (torch.linalg.inv(w) @ x.T).T
            return y, -torch.sum(self.du)
