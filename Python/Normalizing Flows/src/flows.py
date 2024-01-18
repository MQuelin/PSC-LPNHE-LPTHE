import torch.nn as nn

from bijective_transforms import *
from misc_transforms import *

class SimplePlanarNF(nn.Module):
    """
    A simple Normalizing Flow where each layer is a Tanh Planar Flow
    """
    def __init__(self, flow_length, data_dim):
        super().__init__()

        self.layers = nn.Sequential()
        for k in range(flow_length):
            self.layers.add_module(f'Module_{k}', TanhPlanarFlow(data_dim))

    def forward(self, z):
        log_jacobians = 0
        for layer in self.layers:
            z, log_jacobian = layer(z)
            log_jacobians += log_jacobian
        return z, log_jacobians

class SimpleAdditiveNF(nn.Module):
    """
    A simple Normalizing Flow where each layer is an Additive Layer
    """
    def __init__(self, flow_length, data_dim):
        super().__init__()

        n = data_dim//2

        self.layers = nn.Sequential()
        for k in range(flow_length):
            self.layers.add_module(f'Module_{k}', AdditiveCouplingLayer(data_dim,
                                                                        n=n,
                                                                        m=MLP([n, 10, 10, data_dim - n]),
                                                                        s=MLP([n, 10, 10, data_dim - n])
                                                                        ))
        
    def forward(self, z, reverse="false"):
        log_jacobians = 0
        for layer in self.layers:
            z, log_jacobian = layer(z, reverse)
            log_jacobians += log_jacobian
        return z, log_jacobians