import torch
import torch.nn as nn
from numpy.random import permutation
from misc_transforms import MLP

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
        -n: int: number coordinates of initial vector that will be preserved through the identity transform
        -m: lambda (x: Torch.tensor of size(n) -> Torch.tensor of size(data_dim - n)), point wise operation function,
            operating on the last dim of a Torch.tensor
            can be a trainable neural network
        -s: lambda (x: Torch.tensor of size(n) -> Torch.tensor of size(data_dim - n)), point wise operation function,
            operating on the last dim of a Torch.tensor
            can be a trainable neural network

    Forward mapping process without scaling:
        z = (z1,z2) where size of z1 is (n), the split is the result of a random permutation. This permutation is a constant of each additive coupling layer instance
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

        self.permutation_tensor = torch.LongTensor(permutation([i for i in range(data_dim)]))
        self.reverse_permutation_tensor = torch.LongTensor(permutation([0 for i in range(data_dim)]))
        for i in range(data_dim):
            self.reverse_permutation_tensor[self.permutation_tensor[i]] = i

    
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

            # Permute Tensor acording to the permutation describded in the permutation tensor
            z = z[:,self.permutation_tensor]

            #Perform operation
            (z1,z2) = torch.split(z, (self.n, data_dim - self.n), dim = 1)

            scaling_vector = torch.exp(self.s(z1))

            x1 = z1
            x2 = torch.exp(scaling_vector) * z2 + self.m(z1) 
            x = torch.concat((x1,x2), dim=1)

            log_det = torch.log(scaling_vector.sum(1))

            return x, log_det
        else:

            # Permute Tensor acording to the permutation describded in the permutation tensor
            z = z[:,self.reverse_permutation_tensor]

            (z1,z2) = torch.split(z, (self.n, data_dim - self.n), dim = 1)

            scaling_vector = torch.exp(self.s(z1))

            x1 = z1
            x2 = (z2 - self.m(z1))/scaling_vector
            x = torch.concat((x1,x2), dim=1)

            log_det = - torch.log(scaling_vector.sum(1))

            return x, log_det


class ConditionalAffineCouplingLayer(nn.Module):
    """
    Module used to apply an additive coupling layer transform, conditioned by a vector (see below) to a batch of n-dimmensionnal data points
    This is a modified version of AdditiveCouplingLayer used to train models with conditioning
    ------------
    Forward mapping process with scaling and conditioning: (here @ is the Hadamar matrix product)

        z = (z1,z2) where size of z1 is (n)
        
        x1 = z1 @ exp(s1(z2,c)) + m1(z2,c)
        x2 = z2 @ exp(s2(x1,c)) + m2(x1,c)
    
    Reverse mapping process with scaling and conditioning: (here @ is the Hadamar matrix product)

        x = (x1,x2) where size of x1 is (data_dim - n)
        
        z2 = (x2 - m2(x1,c)) @ ( 1/exp(s2(x1,c)) )
        z1 = (x1 - m1(z2,c)) @ ( 1/exp(s1(z2,c)) )
                
    see for reference:  https://deepgenerativemodels.github.io/notes/flow/
                        https://arxiv.org/abs/1907.02392
                        https://arxiv.org/abs/1911.02052
    """

    def __init__(self, input_dim: int, output_dim: int, n: int, m1, m2, s1, s2):
        """
        Arguments:
            - input_dim: int: dimension of the input or 'label'
            - output_dim: int: dimension of the data
            - n: int: number coordinates of initial vector that will be preserved through the identity transform
            - m1: lambda x: Torch.tensor of size (data_dim - n + input_dim) -> Torch.tensor of size(n), point wise operation function,
                operating on the last dim of a Torch.tensor
                can be a trainable neural network
            - m2: lambda x: Torch.tensor of size (n + input_dim) -> Torch.tensor of size(data_dim - n), point wise operation function,
                operating on the last dim of a Torch.tensor
                can be a trainable neural network
            - s1: lambda x: Torch.tensor of size (data_dim - n + input_dim) -> Torch.tensor of size(n), point wise operation function,
                operating on the last dim of a Torch.tensor
                can be a trainable neural network
            - s2: lambda x: Torch.tensor of size (n + input_dim) -> Torch.tensor of size(data_dim - n), point wise operation function,
                operating on the last dim of a Torch.tensor
                can be a trainable neural network

    
        """
        super().__init__()

        assert n<=output_dim

        self.n = n
        self.m1 = m1
        self.m2 = m2
        self.s1 = s1
        self.s2 = s2

        self.permutation_tensor = torch.LongTensor(permutation([i for i in range(output_dim)]))
        self.reverse_permutation_tensor = torch.LongTensor([0 for i in range(output_dim)])
        for i in range(output_dim):
            self.reverse_permutation_tensor[self.permutation_tensor[i]] = i


    def forward(self, c, z, reverse="false"):
        """
        Performs transform on axis 1 of z
        
        Arguments:
            -c: Torch.tensor of size (m, input_dim) where m is batch size and data_dim is dim of input/condition vectors
            -z: Torch.tensor of size (m, data_dim) where m is batch size and data_dim is dim of data vectors

        Returns: 
            -x: Torch.tensor of size (m, data_dim) where x is the result of the transform
            -log_det: the log of the absolute value of the determinent of the Jacobian of fθ evaluated in z
        """
        data_dim = z.shape[-1]

        if not reverse:

            # Permute Tensor acording to the permutation describded in the permutation tensor
            z = z[:,self.permutation_tensor]

            #Split tensor
            (z1,z2) = torch.split(z, (self.n, data_dim - self.n), dim = 1)

            z2c = torch.concat((z2,c), dim=1)
            scaling_vector_1 = self.s1(z2c)
            x1 = z1 * torch.exp(scaling_vector_1) + self.m1(z2c)

            x1c = torch.concat((x1,c), dim=1)
            scaling_vector_2 = self.s2(x1c)
            x2 = z2 * torch.exp(scaling_vector_2) + self.m2(x1c)

            x = torch.concat((x1,x2), dim=1)

            log_det = scaling_vector_1.sum(1) + scaling_vector_2.sum(1)

            return x, log_det
        else:

            #Split tensor
            (x1,x2) = torch.split(z, (self.n, data_dim - self.n), dim = 1)


            x1c = torch.concat((x1,c), dim=1)
            scaling_vector_1 = self.s2(x1c)
            z2 = (x2 - self.m2(x1c)) / torch.exp(scaling_vector_1)

            z2c = torch.concat((z2,c), dim=1)
            scaling_vector_2 = self.s1(z2c)
            z1 = (x1 - self.m1(z2c)) / torch.exp(scaling_vector_2)

            z = torch.concat((z1,z2), dim=1)

            log_det = - scaling_vector_1.sum(1) - scaling_vector_2.sum(1)

            # Permute Tensor acording to the reverse permutation describded in the permutation tensor
            z = z[:,self.reverse_permutation_tensor]

            return z, log_det

class AutoRegressiveLayer(nn.Module):
    """
    Module used to apply an auto regressive transform
    ------------
    Forward mapping process with scaling and conditioning:

        z = (z1, ..., zn)
        xi = zi * si(z1,z2,...,z_i-1) + mi(z1,z2,...,z_i-1)
        log(det(jacobian)) = sum(si)

        this process is done in parallel and is the choosen sampling process:
    
    Reverse mapping process :

        x = (x1, ...,xn)
        zi = (xi - mi(z1,z2,...,z_i-1)) / si(z1,z2,...,z_i-1)
        log(det(jacobian)) = -sum(si)

        this process must be done sequentially and is used for training
                
    see for reference:  https://lilianweng.github.io/posts/2018-10-13-flow-models/
                        https://arxiv.org/pdf/1606.04934.pdf
    """

    def __init__(self, dim: int):
        """
        Arguments:
            - dim: int: dimension of the vectors that will go through the layer/flow

        """
        super().__init__()
        self.dim = dim
        self.s_list= nn.ParameterList()
        self.m_list= nn.ParameterList()
        for k in range(dim-1):
            self.s_list.append(MLP([k+1,1],activation_layer=nn.Tanh()))
            self.m_list.append(MLP([k+1,1],activation_layer=nn.Tanh()))
        
        # As the first transforms are not linear per se as they take in no input we declare them as a scalars here
        self.s0 = nn.Parameter(torch.zeros(1))
        self.m0 = nn.Parameter(torch.zeros(1))

        self.permutation_tensor = torch.LongTensor(permutation([i for i in range(dim)]))
        self.reverse_permutation_tensor = torch.LongTensor([0 for i in range(dim)])
        for i in range(dim):
            self.reverse_permutation_tensor[self.permutation_tensor[i]] = i
        
    def forward(self, z, reverse="false"):
        x = torch.zeros_like(z)
        log_jac_det = torch.zeros_like(z[::,0])

        # IMPORTANT TODO
        # Render the forward process parallel and not sequential to acheive better computational efficiency
        if not reverse:
            # Permute Tensor acording to the permutation describded in the permutation tensor
            z = z[:,self.permutation_tensor]
            for i in range(self.dim):

                # xi = zi * si(z1,z2,...,z_i-1) + mi(z1,z2,...,z_i-1)
                if i==0:
                    scaling_vector = self.s0
                    x[::,i] = z[::,i] * torch.exp(scaling_vector) + self.m0
                    log_jac_det += scaling_vector

                else :
                    s = self.s_list[i-1]
                    m = self.m_list[i-1]
                    scaling_vector = s(z[::,0:i])
                    x[::,i] = z[::,i] * torch.exp(scaling_vector).squeeze(1) + m(z[::,0:i]).squeeze(1)

                    log_jac_det += scaling_vector.squeeze(1)

        else:
            # Permute Tensor acording to the permutation describded in the permutation tensor
            z = z[:,self.reverse_permutation_tensor]
            for i in range(self.dim):
                # zi = (xi - mi(z1,z2,...,z_i-1)) / si(z1,z2,...,z_i-1)
                # To avoid changing the arguments name we simply replace x by z in the above formula
                if i==0:
                    scaling_vector = self.s0
                    x[::,i] = (z[::,i] - self.m0) * torch.exp(-scaling_vector)
                    log_jac_det += -scaling_vector
                
                else :
                    s = self.s_list[i-1]
                    m = self.m_list[i-1]
                    scaling_vector = s(x[::,0:i])
                    x[::,i] = (z[::,i] - m(z[::,0:i]).squeeze(1)) * torch.exp(-scaling_vector).squeeze(1)

                    log_jac_det += -scaling_vector.squeeze(1)

        
        return x, log_jac_det