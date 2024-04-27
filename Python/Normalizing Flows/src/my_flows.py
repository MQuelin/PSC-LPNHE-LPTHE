import torch.nn as nn

import numpy as np
from my_bijective_transforms import *
from bijective_transforms import *
from misc_transforms import *    

class ConditionalQS(nn.Module):
    """
    A flow utilizing conditional quadratic splines
    """
    def __init__(self, flow_length, input_dim, output_dim, K, device):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device=device

        n = output_dim//2

        self.layers = nn.Sequential()
        for k in range(flow_length):
            self.layers.add_module(f'Module_{k}', QuadraticsSpline(input_dim=input_dim,
                                                                                 output_dim=output_dim,
                                                                                 n=n,
                                                                                 K=K,
                                                                                 s=MLP([output_dim - n + input_dim, 10, 20,50,100, n*(2*K+1)],device=self.device)
                                                                                ))
        
    def forward(self, c, z, reverse="false"):
        log_jacobians = 0
        if reverse:
            z = torch.sigmoid(10*z)
            log_jacobians=(z*(1-z)).sum(1)
        for layer in self.layers:
            z, log_jacobian = layer(c, z, reverse)
            log_jacobians += log_jacobian
        if not(reverse):
            z=torch.logit(z)
        return z, log_jacobians

    def sample(self, c):

        batch_size = c.shape[0]
        c = c.to(self.device)
   
        dummy_variable = torch.randn(size=(batch_size,self.output_dim-self.input_dim)).to(self.device)

        if self.input_dim ==1:
            c = c.unsqueeze(dim=1)

        z = torch.concat((c,dummy_variable), dim=1)

        return self.forward(c, z, reverse=False)