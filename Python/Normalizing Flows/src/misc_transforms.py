import torch
import torch.nn as nn

"""
File containing transforms that can be passed as arguments to some of the coupling layers present in bijective_transforms.
"""

class Scale(torch.nn.Module):
    def __init__(self, a: float):
        """
        a: Scalar such that Scale(X) returns aX
        """
        super().__init__()
        self.a = a

    def forward(self, X):
        return self.a*X
    



class MLP(nn.Module):
    """
    Simple Multi Layered Perceptron with ReLU activation layers and a customizable final activation layer.
    """

    def __init__(self, features_dim_list, activation_layer = nn.ReLU(), scaling_factor = 1.0, device="cuda"):
        """
        Arguments:
            -features_dim_list: list of dimensions of each layer. first(resp. last) int in the list is the input(resp. output) dimension of the transform
        """
        super().__init__()

        self.transform = nn.Sequential()

        self.transform.add_module(f'module_{0}_1', nn.Linear(features_dim_list[0], features_dim_list[1], device=device))
        self.transform.add_module(f'module_{0}_2', nn.ReLU())

        for k in range(len(features_dim_list)-2):
            self.transform.add_module(f'module_{k+1}_1', nn.Linear(features_dim_list[k+1], features_dim_list[k+2], device=device))
            self.transform.add_module(f'module_{k+1}_2', activation_layer)
        self.transform.add_module(f'Final scaling', Scale(scaling_factor))

        
    def forward(self, z):
        """
        Performs mlp transform on z
        
        Arguments:
            -z: Torch.tensor of size (m, data_dim) where m is batch size and data_dim is dim of input data vectors

        Returns: 
            -x: Torch.tensor of size (m, output_dim)
        """
        
        return self.transform(z)


class GaussianLayer(nn.Module):
    """
    A neural network applying a gaussian transformation over a batch of input vectors of shape [batch_size,dim]
    """
    
    def __init__(self, in_features, out_features, device='cuda'):
        """
        Arguments:
            -in_features number of dimensions of input vectors
            -out_features number of dimensiosn of output vectors
        """
        super().__init__()
        # ( z * (z@cov_batch) ).sum(dim=2).T
        self.diag_params = nn.Parameter(torch.randn(size=(out_features,in_features))).to(device)
        self.mean_params = nn.Parameter(5*torch.randn(size=(out_features, in_features,1))).to(device)
        self.scale_params = nn.Parameter(torch.randn(out_features)).to(device)

        self._clamp_func1 = nn.Sigmoid()
        self._clamp_func2 = nn.Tanh()
    
    def clamped_diag_params(self):
        floor = 0.1
        ceil = 1

        delta = (ceil - floor)
        return floor + delta*self._clamp_func1(self.diag_params)
    
    def clamped_scale_params(self):
        return self._clamp_func2(self.scale_params)

    def forward(self, z):
        """
        performs gaussian transform on z

        Arguments:
            -z: Torch.tensor of size (m, data_dim) where m is batch size and data_dim is dim of input data vectors

        Returns: 
            -x: Torch.tensor of size (m, output_dim)
        """
        cov_batch = torch.stack([torch.diag(self.clamped_diag_params()[k]) for k in range(self.diag_params.shape[0])])

        z_cov_z = ( z * (z@cov_batch) ).sum(dim=2).T
        mean_cov_mean = ( self.mean_params.transpose(1,2) @ (cov_batch@self.mean_params) ).squeeze(1).squeeze(1)
        z_cov_mean = ( z @ (cov_batch@self.mean_params) ).squeeze(2).T

        scaling_vector = self.clamped_scale_params()

        return scaling_vector * torch.exp(- z_cov_z - mean_cov_mean + 2*z_cov_mean)
        

class GAU(nn.Module):
    """
    Gaussian activation unit
    """
    def __init__(self, in_features, n, out_features, device) -> None:
        super().__init__()

        self.transform = nn.Sequential()
        self.transform.add_module('Gaussian Layer', GaussianLayer(in_features, n, device))
        self.transform.add_module('Linear Layer', nn.Linear(n, out_features, bias=False, device=device))
    
    def forward(self, z):

        return self.transform(z)



class InvertibleMapping(nn.Module):
    """
    invertible mapping object that applies a linear transform to a batch of vectors
    i.e X = A.Z and X = A_inv.Z
    furthermore |det(A)| = |det(A_inv)| = 1, so that the transformation preserves volume
    """
    def __init__(self, dim) -> None:
        super().__init__()
        A = torch.randn(dim,dim)
        # we normalize the matrix here
        # we do so twice to try and ensure proper invertibility
        for k in range(2):
            det = torch.abs(torch.linalg.det(A))
            A = A/(det**(1/dim))
        
        
        self.A = A
        self.A.requires_grad = False
        self.A_inv = torch.linalg.inv(self.A)
        self.A_inv.requires_grad = False
        self.device = 'cpu'
    
    def forward(self, z, reverse = False):
        device = z.device
        if self.device != device:
            self.A = self.A.to(device)
            self.A_inv = self.A_inv.to(device)
        z = torch.transpose(z,0,1)
        if not reverse:
            return torch.transpose(self.A@z,0,1)
        else:
            return torch.transpose(self.A_inv@z,0,1)

class DampenedRoot(nn.Module):
    """
    Applies a dampened root function elementwise
    x = x(1-exp(-1/x^2)) + exp(-1/x^2) * sign(x) * (|x|) ^ (1/n) where n is an integer
    """

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        # x is a matrix batch x hidden dim
        a = torch.exp(-(torch.pow(x,-2)))
        return x*(1-a) + a*torch.sign(x)*torch.pow(torch.abs(x),1/self.n)

class ReLogU(nn.Module):
    """
    Applies a 'rectified logarithmic unit' function elementwise
    x = sign(x)*log(1+|x|)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x is a matrix batch x hidden dim
        return torch.sign(x)*torch.log(1+torch.abs(x))

