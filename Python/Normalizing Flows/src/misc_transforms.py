import torch
import torch.nn as nn

"""
File containing transforms that can be passed as arguments to some of the coupling layers present in bijective_transforms.
"""

class MLP(nn.Module):
    """
    Simple Multi Layered Perceptron with ReLU activation layers
    """

    def __init__(self, input_dim, layer_dim_list):
        """
        Arguments:
            -input_dim: dimension of input
            -layer_dim_list: list of dimensions of each layer. Last int in the list is the output dimension of the transform
        """

        self.transform = nn.Sequential()

        for n in layer_dim_list:
            self.transform.add_module(nn.Linear(input_dim, n))
            self.transform.add_module(nn.ReLU())
            input_dim = n
        
    def forward(self, z):
        """
        Performs mlp transform on z
        
        Arguments:
            -z: Torch.tensor of size (m, data_dim) where m is batch size and data_dim is dim of input data vectors

        Returns: 
            -x: Torch.tensor of size (m, output_dim)
        """

        x = self.transform(z)

        return x