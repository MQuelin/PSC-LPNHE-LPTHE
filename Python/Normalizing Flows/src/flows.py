import torch.nn as nn

from bijective_transforms import *
from misc_transforms import *
from LU_class_dev import LULayer

from maf_layer import MAFLayer

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
    Bugged
    A simple Normalizing Flow where each layer is an Additive Layer
    """
    def __init__(self, flow_length, data_dim):
        super().__init__()

        self.output_dim = data_dim
        n = data_dim//2

        self.layers = nn.ModuleList()
        for k in range(flow_length):
            self.layers.append(AdditiveCouplingLayer(data_dim,
                                                            n=n,
                                                            m=MLP([n, 10, 10, data_dim - n]),
                                                            s=MLP([n, 10, 10, data_dim - n])
                                                            ))
        self.flow_length = flow_length
    
    def forward(self, z, reverse="false"):
        log_jacobians = 0
        if not reverse:
            for k in range(self.flow_length):
                z, log_jacobian = self.layers[k](z, reverse)
                log_jacobians += log_jacobian
        else:
            for k in range(self.flow_length):
                z, log_jacobian = self.layers[-1-k](z, reverse)
                log_jacobians += log_jacobian            
        return z, log_jacobians
    
    def sample(self, n):
        """
        Takes in an int n representing the desired amout of samples
        Returns a tensor of shape [n,d] where d is the dimension of the output vectors of the model
        """
        z = torch.randn(size=(n,self.output_dim)).to(self.device)

        samples, _log_dets = self.forward(z,reverse=False)

        # We detach the output as we should not require gradients after this step
        return samples.cpu().detach()

class ConditionalNF(nn.Module):
    """
    A flow utilizing conditional affine coupling layers
    MLPs are used for applying the affine coupling layer
    The default shape for the MLP is four layers with 10 weights in the center and the appropriate weights to maintain valid input and output dimensions
    """

    def __init__(self, flow_length, input_dim, output_dim, MLP_shape_list = [10,10], device='cuda'):
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

        self.layers = nn.ModuleList()
        self.flow_length = flow_length
        for k in range(flow_length):
            shape_list_1 = [output_dim - n + input_dim] + MLP_shape_list + [n]
            shape_list_2 = [n + input_dim] + MLP_shape_list + [output_dim - n]
            m1 = MLP(shape_list_1, activation_layer=nn.Tanh())
            m2 = MLP(shape_list_2, activation_layer=nn.Tanh())
            s1 = MLP(shape_list_1, activation_layer=nn.Tanh())
            s2 = MLP(shape_list_2, activation_layer=nn.Tanh())
            self.layers.append(ConditionalAffineCouplingLayer(  input_dim=input_dim,
                                                                output_dim=output_dim,
                                                                n=n,
                                                                m1=m1,m2=m2,s1=s1,s2=s2))
        
    def forward(self, c, z, reverse="false"):
        log_jacobians = 0
        if not reverse:
            for k in range(self.flow_length):
                z, log_jacobian = self.layers[k](c, z, reverse)
                log_jacobians += log_jacobian
        else:
            for k in range(self.flow_length):
                z, log_jacobian = self.layers[-1-k](c, z, reverse)
                log_jacobians += log_jacobian            
        return z, log_jacobians

    def sample(self, c):

        batch_size = c.shape[0]
        c = c.to(self.device)
   
        dummy_variable = torch.randn(size=(batch_size,self.output_dim-self.input_dim)).to(self.device)

        if self.input_dim ==1:
            c = c.unsqueeze(dim=1)

        z = torch.concat((c,dummy_variable), dim=1)

        return self.forward(c, z, reverse=False)
    
class ImageConditionalNF(nn.Module):
    """
    A flow utilizing conditional affine coupling layers for image generation. Images must be both square and black and white i.e with a single channel
    image.size() = [BATCH_SIZE, 1 , WIDTH, WIDTH]
    """
    def __init__(self, flow_length, input_dim, image_width, MLP_shape_list = [10,10], device='cuda'):
        """
        """
        super().__init__()
        self.input_dim = input_dim
        self.width = image_width
        self.output_dim = image_width**2
        self.device=device

        n = self.output_dim//2

        self.layers = nn.ModuleList()
        self.flow_length = flow_length
        for k in range(flow_length):
            shape_list_1 = [self.output_dim - n + input_dim] + MLP_shape_list + [n]
            shape_list_2 = [n + input_dim] + MLP_shape_list + [self.output_dim - n]
            m1 = MLP(shape_list_1)
            m2 = MLP(shape_list_2)
            s1 = MLP(shape_list_1)
            s2 = MLP(shape_list_2)
            self.layers.append(ConditionalAffineCouplingLayer(  input_dim=input_dim,
                                                                output_dim=self.output_dim,
                                                                n=n,
                                                                m1=m1,m2=m2,s1=s1,s2=s2))
        
    def forward(self, c, z, reverse="false"):
        log_jacobians = 0
        if not reverse:
            for k in range(self.flow_length):
                z, log_jacobian = self.layers[k](c, z, reverse)
                log_jacobians += log_jacobian
        else:
            for k in range(self.flow_length):
                z, log_jacobian = self.layers[-1-k](c, z, reverse)
                log_jacobians += log_jacobian            
        return z, log_jacobians

    def sample(self, c):
        """
        Takes in a condition tensor c such that c.shape = (Batch_size, number_of_conditions) and returns a square image tensor i such that
        i.shape = (Batch_size, 1, width, width) 
        """

        batch_size = c.shape[0]
        c = c.to(self.device)

        dummy_variable = torch.randn(size=(batch_size,self.output_dim-self.input_dim)).to(self.device)

        z = torch.concat((c,dummy_variable), dim=1)

        output, _log_jacobians = self.forward(c, z, reverse=False)

        #Converting vector to image
        output = output.view(output.shape[0], self.width, self.width)
        #unsqueezing vector to add a single channel
        output = torch.unsqueeze(output,1)

        return output   

class ConditionaMAF(nn.Module):

    def __init__(self, flow_length, input_dim, output_dim, device='cuda'):
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device=device

        self.layers = nn.ModuleList()
        self.flow_length = flow_length

        for k in range(flow_length):
            self.layers.append(MAFLayer(output_dim, input_dim))
        
    def forward(self, c, z, reverse=False):
        log_jacobians = 0

        for k in range(self.flow_length):
            z, log_jacobian_M = self.layers[k](z, c)
            log_jacobians += log_jacobian_M

        return z, log_jacobians

    def sample(self, c):

        batch_size = c.shape[0]
        c = c.to(self.device)

        dummy_variable = torch.randn(size=(batch_size,self.output_dim-self.input_dim)).to(self.device)

        z = torch.concat((c,dummy_variable), dim=1)

        return self.forward(c, z, reverse=False)
    
class SimpleIAF(nn.Module):
    def __init__(self, flow_length, output_dim, device='cuda'):
        super().__init__()
        
        self.flow_length = flow_length
        self.output_dim = output_dim
        self.device = device

        self.layers = nn.ModuleList()
        for k in range(flow_length):
            self.layers.append(AutoRegressiveLayer(output_dim))
    
    def forward(self, z, reverse="false"):
        log_jacobians = 0
        if not reverse:
            for k in range(self.flow_length):
                z, log_jacobian = self.layers[k](z, reverse)
                log_jacobians += log_jacobian
        else:
            for k in range(self.flow_length):
                z, log_jacobian = self.layers[-1-k](z, reverse)
                log_jacobians += log_jacobian            
        return z, log_jacobians

    def sample(self, n):
        """
        Takes in an int n representing the desired amout of samples
        Returns a tensor of shape [n,d] where d is the dimension of the output vectors of the model
        """
        z = torch.randn(size=(n,self.output_dim)).to(self.device)

        samples, _log_dets = self.forward(z,reverse=False)

        # We detach the output as we should not require gradients after this step
        return samples.cpu().detach()

class SimpleIAF2(nn.Module):
    def __init__(self, flow_length, output_dim, device='cuda'):
        super().__init__()
        
        self.flow_length = flow_length
        self.output_dim = output_dim
        self.device = device

        self.layers = nn.ModuleList()
        for k in range(flow_length):
            self.layers.append(AutoRegressiveLayer2(output_dim).to(device))
    
    def forward(self, z, reverse="false"):
        log_jacobians = 0
        if not reverse:
            for k in range(self.flow_length):
                z, log_jacobian = self.layers[k](z, reverse)
                log_jacobians += log_jacobian
        else:
            for k in range(self.flow_length):
                z, log_jacobian = self.layers[-1-k](z, reverse)
                log_jacobians += log_jacobian            
        return z, log_jacobians

    def sample(self, n):
        """
        Takes in an int n representing the desired amout of samples
        Returns a tensor of shape [n,d] where d is the dimension of the output vectors of the model
        """
        z = torch.randn(size=(n,self.output_dim)).to(self.device)

        samples, _log_dets = self.forward(z,reverse=False)

        # We detach the output as we should not require gradients after this step
        return samples.cpu().detach()

class CIAF(nn.Module):
    def __init__(self, flow_length, input_dim, output_dim, device='cuda'):
        super().__init__()
        
        self.flow_length = flow_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.layers = nn.ModuleList()
        for k in range(flow_length):
            self.layers.append(ConditionalAutoRegressiveLayer2(input_dim, output_dim))
    
    def forward(self, c, z, reverse="false"):
        log_jacobians = 0
        if not reverse:
            for k in range(self.flow_length):
                z, log_jacobian = self.layers[k](c, z, reverse)
                log_jacobians += log_jacobian
        else:
            for k in range(self.flow_length):
                z, log_jacobian = self.layers[-1-k](c, z, reverse)
                log_jacobians += log_jacobian            
        return z, log_jacobians

    def sample(self, c):
        """
        Takes in a batch of conditions c
        Returns a tensor of samples
        """
        batch_size = c.shape[0]
        c = c.to(self.device)

        dummy_variable = torch.randn(size=(batch_size,self.output_dim-self.input_dim)).to(self.device)
        
        if self.input_dim ==1:
            c = c.unsqueeze(dim=1)

        z = torch.concat((c,dummy_variable), dim=1)

        return self.forward(c, z, reverse=False)
    
class TestLU(nn.Module):
    def __init__(self, flow_length, dim):
        super().__init__()

        self.dim = dim
        self.layers = nn.ModuleList()

        for k in range(flow_length):
            self.layers.append(LULayer(dim))

    def forward(self, z, reverse=False):
        log_jacobians = 0
        for layer in self.layers:
            z, log_jacobian = layer(z)
            log_jacobians += log_jacobian
        return z, log_jacobians