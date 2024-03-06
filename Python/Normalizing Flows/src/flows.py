import torch.nn as nn

import numpy as np
from scipy.stats import multivariate_normal

from bijective_transforms import *
from misc_transforms import *


class SimplePlanarNF(nn.Module):
    """
    Deprecated

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

class ConditionalNF(nn.Module):
    """
    A flow utilizing conditional affine coupling layers
    MLPs are used for applying the affine coupling layer
    The default shape for the MLP is four layers with 10 weights in the center and the appropriate weights to maintain valid input and output dimensions
    """

    def __init__(self, flow_length, input_dim, output_dim,MLP_shape_list = [10,10], device='cuda'):
        """
        MLP_shape_list: used to modify the shape of the MLPs used in the affine coupling layers;
                        the length of the list is the number of inner layers;
                        the int at idex i is the number of weights of the layer of index i in the MLP
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device=device

        n = output_dim//2

        self.layers = nn.Sequential()
        for k in range(flow_length):
            shape_list_1 = [output_dim - n + input_dim] + MLP_shape_list + [n]
            shape_list_2 = [n + input_dim] + MLP_shape_list + [output_dim - n]
            m1 = MLP(shape_list_1)
            m2 = MLP(shape_list_2)
            s1 = MLP(shape_list_1)
            s2 = MLP(shape_list_2)
            self.layers.add_module(f'Module_{k}', ConditionalAffineCouplingLayer(input_dim=input_dim,
                                                                                 output_dim=output_dim,
                                                                                 n=n,
                                                                                 m1=m1,m2=m2,s1=s1,s2=s2
                                                                                ))
        
    def forward(self, c, z, reverse="false"):
        log_jacobians = 0
        for layer in self.layers:
            z, log_jacobian = layer(c, z, reverse)
            log_jacobians += log_jacobian
        return z, log_jacobians

    def sample(self, c):

        batch_size = c.shape[0]
        c = c.to(self.device)

        gaussian = multivariate_normal(cov=np.eye(self.output_dim-self.input_dim))
        dummy_variable = torch.tensor(gaussian.rvs(size = batch_size)).to(self.device).to(torch.float32)

        z = torch.concat((c,dummy_variable), dim=1)

        return self.forward(c, z, reverse=False)