import torch
import torch.nn as nn
import math

#TODO fix function output shape

def gaussian_log_pdf(z):
    """
    Arguments:
        - z: a batch of m data points (size: m x data_dim)

    Returns:
        - torch.tensor of size (m) containing the sum of the Normal distribution's PDF evaluated at the data points in z
    """
    
    a = z**2

    b = -.5 * torch.log( (math.pi) * 2 + a ** 2).sum(1)

    return b