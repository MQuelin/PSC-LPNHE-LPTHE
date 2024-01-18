import torch
import torch.nn as nn

"""
File containing transforms that can be passed as arguments to some of the coupling layers present in bijective_transforms.
"""

class MLP(nn.Module):
    """
    Simple Multi Layered Perceptron with ReLU activation layers
    """

    def __init__(self, features_dim_list):
        """
        Arguments:
            -features_dim_list: list of dimensions of each layer. first(resp. last) int in the list is the input(resp. output) dimension of the transform
        """
        super().__init__()

        self.transform = nn.Sequential()

        for k in range(len(features_dim_list)-1):
            self.transform.add_module(f'module_{k}_1', nn.Linear(features_dim_list[k], features_dim_list[k+1]))
            self.transform.add_module(f'module_{k}_2', nn.ReLU())
        
    def forward(self, z):
        """
        Performs mlp transform on z
        
        Arguments:
            -z: Torch.tensor of size (m, data_dim) where m is batch size and data_dim is dim of input data vectors

        Returns: 
            -x: Torch.tensor of size (m, output_dim)
        """
        
        return self.transform(z)