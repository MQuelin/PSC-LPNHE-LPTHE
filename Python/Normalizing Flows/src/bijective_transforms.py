import torch
import torch.nn as nn

"""
File containing different classes used as bijective transforms that can be composed together to create normalizing flows
"""

class TanhPrime(nn.Module):

    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, z):
        return 1 - self.tanh(z)**2

class TanhPlanarFlow(nn.Module):
    """
    Module used to apply a planar flow transform (see below) to a batch of n-dimmensionnal data points
    ------------
    Fields:
        -u,w :      Learning parameters, n dimmentionnal real vectors
        -b :        Learning parameter, real scalar
        -h :        the Torch.Tanh function
        -h_prime :  the derivative of Tanh

    Planar flow transform: x = fθ(z) = z + u * h (Transpose(w) * z + b)
    Absolute value of determinant of Jacobian: det = 1 + h'(Transpose(w) * z + b) * Transpose(u) * w
    Here h is set as equal to tanh
    For the transform to remain bijective, u must be constrained.
    
    see for reference: https://deepgenerativemodels.github.io/notes/flow/
    """

    def __init__(self,data_dim: int):
        """
        Arguments:
            - data_dim: int: dimension of the data distribution
        """
        super().__init__()

        self.u = nn.Parameter(torch.rand(data_dim,1))
        self.w = nn.Parameter(torch.rand(data_dim,1))
        self.b = nn.Parameter(torch.rand(1))
        self.h = nn.Tanh()
        self.h_prime = TanhPrime()

    def __constrained__u(self):
        """
        Constrains the parameter u to ensure invertibility of the planar flow transform
        """
        wu = torch.matmul(self.w.T, self.u)
        m = lambda x: -1 + torch.log(1 + torch.exp(x))
        return self.u + (m(wu) - wu) * (self.w / (torch.norm(self.w) ** 2 + 1e-15))

    def forward(self, z):
        """
        Constrains u and performs planar flow transform on axis 1 of tensor z
        
        Arguments:
            -z: Torch.tensor of size (m, data_dim) where m is batch size and data_dim is dim of data vectors

        Returns: 
            -x: Torch.tensor of size (m, data_dim) where x = fθ(z) = z + u * h (Transpose(w) * z + b)
            -log_det: the log of the absolute value of the determinent of the Jacobian of fθ evaluated in z
        """

        constrained_u = self.__constrained__u()

        hidden_units = torch.matmul(z, self.w) + self.b
        x = z + constrained_u.T * self.h(hidden_units)

        psi = torch.matmul(constrained_u.T, self.w)
        log_det = torch.log((1 + psi*self.h_prime(hidden_units)).abs() + 1e-15)

        return x, log_det


class AdditiveCouplingLayer(nn.Module):
    """
    Module used to apply an additive coupling layer transform (see below) to a batch of n-dimmensionnal data points
    ------------
    Fields:
        -n: int: numbers of first coordinates of initial vector that will be preserved through the identity transform
        -m: lambda (x: Torch.tensor of size(n) -> Torch.tensor of size(data_dim - n)), point wise operation function,
            operating on the last dim of a Torch.tensor
            can be a trainable neural network
        -s: lambda (x: Torch.tensor of size(n) -> Torch.tensor of size(data_dim - n)), point wise operation function,
            operating on the last dim of a Torch.tensor
            can be a trainable neural network

    Forward mapping process without scaling:
        z = (z1,z2) where size of z1 is (n)
        x1 = z1
        x2 = z2 + m(z1)         (this is why the layer is said to be additive)
    
    Forward mapping process with scaling:
        z = (z1,z2) where size of z1 is (n)
        x1 = z1
        x2 = exp(s(z1)) @ z2 + m(z1)         (where @ is the Hadamar matricial product)
                
    see for reference: https://deepgenerativemodels.github.io/notes/flow/
    """

    def __init__(self, data_dim: int, n: int, m, s):
        """
        Arguments:
            - data_dim: int: dimension of the data distribution
        """
        super().__init__()

        assert n<=data_dim

        self.n = n
        self.m = m
        self.s = s
    
    def forward(self, z, reverse="false"):
        """
        Performs transform on axis 1 of z
        
        Arguments:
            -z: Torch.tensor of size (m, data_dim) where m is batch size and data_dim is dim of data vectors

        Returns: 
            -x: Torch.tensor of size (m, data_dim) where x is the result of the transform
            -log_det: the log of the absolute value of the determinent of the Jacobian of fθ evaluated in z
        """
        data_dim = z.shape[-1]

        if not reverse:

            # Permute Tensor first element
            permutation_tensor = torch.LongTensor([5,0,1,2,3,4])
            z = z[:,permutation_tensor]

            #Perform operation
            (z1,z2) = torch.split(z, (self.n, data_dim - self.n), dim = 1)

            scaling_vector = torch.exp(self.s(z1))

            x1 = z1
            x2 = torch.exp(scaling_vector) * z2 + self.m(z1) 
            x = torch.concat((x1,x2), dim=1)

            log_det = scaling_vector.sum(1)

            return x, log_det
        else:

            # Permute Tensor first element
            reverse_permutation_tensor = torch.LongTensor([1,2,3,4,5,0])
            z = z[:,reverse_permutation_tensor]

            (z1,z2) = torch.split(z, (self.n, data_dim - self.n), dim = 1)

            scaling_vector = torch.exp(self.s(z1))

            x1 = z1
            x2 = (z2 - self.m(z1))/scaling_vector
            x = torch.concat((x1,x2), dim=1)

            log_det = - scaling_vector.sum(1)

            return x, log_det


