import torch
from torch import nn
from numpy.random import permutation
import numpy as np

class LULayer(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        # w = special_ortho_group.rvs(dim)
        # print(w)
        # w = np.random.rand(output_dim, input_dim)


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

        self.lu_decomposed = True
    
    def compose_w(p, l, u):
        return torch.mm(torch.mm(p, l), u)
    
  
    def forward(self, x, reverse=False):
        if not reverse :
            w = self.p@self.l@self.u
            #y = torch.mm(x, w)
            y = (w @ x.T).T
            return y, torch.log(torch.abs(torch.det(self.u)))
        else :
            return self.invert(x)
    
    def invert(self, y):
        p_invert = torch.linalg.inv(self.p)
        u_invert = torch.linalg.inv(self.u)
        l_invert = torch.linalg.inv(self.l)

        matrix = u_invert@l_invert@p_invert

        x = matrix@y
        log_det = torch.log(
                torch.abs(torch.det(u_invert)))

        return x, log_det.expand(x.shape[0])
