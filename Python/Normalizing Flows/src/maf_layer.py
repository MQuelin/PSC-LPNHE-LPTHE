from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor


class MAFLayer(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(MAFLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.pa = nn.ParameterList([nn.Parameter(torch.zeros(i+input_dim)) for i in range(output_dim)])
        self.pb = nn.ParameterList([nn.Parameter(torch.zeros(i+input_dim)) for i in range(output_dim)])
    
    def forward(self, X, c, reverse_f=False):
        x = torch.cat((c,X),dim=1).T
        z = x.clone().detach()
        log_det = 0

        if not reverse_f:
            for i in range(self.output_dim):
                a = (self.pa[i] @ x[:i+self.input_dim]).T
                b = (self.pb[i] @ x[:i+self.input_dim]).T
                z[i+self.input_dim] = x[i+self.input_dim] * torch.exp(a) + b
                log_det += a
        else:
            for i in range(self.output_dim):
                a = (self.pa[i] @ z[:i+self.input_dim]).T
                b = (self.pb[i] @ z[:i+self.input_dim]).T
                z[i+self.input_dim] = (x[i+self.input_dim] - b) / torch.exp(a)
                log_det -= a

        z = z[self.input_dim:].T
        return z, log_det
