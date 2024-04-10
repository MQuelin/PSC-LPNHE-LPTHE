import torch
from torch import nn
from numpy.random import permutation
import numpy as np

class TestLULayer(nn.Module):
    def _init_(self, input_dim: int, output_dim: int):
        super()._init_()

        assert(input_dim == output_dim)

        # w = special_ortho_group.rvs(dim)
        # print(w)
        # w = np.random.rand(output_dim, input_dim)

        k = max(output_dim, input_dim)

        """
        w = torch.from_numpy(w)
        p, l, u = torch.linalg.lu(w)
        """

        upper_triangular = torch.triu(torch.randn(k,  input_dim))
        diagonal = torch.diag(torch.abs(torch.randn(k,input_dim))+1e-5)
        u = upper_triangular + diagonal
        for ligne in range(output_dim) :
            for colonne in range(k):
                if ligne <= colonne :
                    l[ligne, colonne].requires_grad = False

        l = torch.tril(torch.randn(output_dim, k))
        l -= torch.diag(torch.diag(l)) - torch.diag(torch.ones(output_dim, k))

        for ligne in range(output_dim) :
            for colonne in range(k):
                if ligne >= colonne :
                    l[ligne, colonne].requires_grad = False

        permutation_tensor = torch.LongTensor(permutation([i for i in range(output_dim)]))

        p_matrix = torch.zeros(output_dim, output_dim)
        for i, p in enumerate(permutation_tensor):
            p_matrix[i, p] = 1

        self.p = p_matrix
        self.l = l
        self.u = u
        nn.Parameter(l,u)
        self.lu_decomposed = True
    
    def compose_w(p, l, u):
        return torch.mm(torch.mm(p, l), u)
  
    def forward(self, x, reverse="false"):
        if not reverse :
            w = self.p@self.l@self.u
            #y = torch.mm(x, w)
            y = w@x
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
