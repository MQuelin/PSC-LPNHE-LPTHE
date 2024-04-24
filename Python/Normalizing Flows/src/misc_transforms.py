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

        for k in range(len(features_dim_list)-1):
            self.transform.add_module(f'module_{k}_1', nn.Linear(features_dim_list[k], features_dim_list[k+1], device=device))
            self.transform.add_module(f'module_{k}_2', activation_layer)
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