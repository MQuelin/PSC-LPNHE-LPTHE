import torch
import torch.nn as nn
from numpy.random import permutation
from misc_transforms import MLP, InvertibleMapping
from flow_utils import *
from torch.nn import functional as F
import torchutils

###Modified code from https://github.com/bayesiains/nflows
        
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3


def unconstrained_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    inverse=False,
    tail_bound=1.0,
    tails="linear",
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
):

    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    print(inside_interval_mask)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    num_bins = unnormalized_widths.shape[-1]

    if tails == "linear":
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
        # print(unnormalized_heights.shape, num_bins - 1)
        #assert unnormalized_heights.shape[-1] == num_bins - 1
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.any(inside_interval_mask):
        outputs[inside_interval_mask], logabsdet[inside_interval_mask] = quadratic_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
        )

    return outputs, logabsdet


def quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
):

    if torch.min(inputs) < left or torch.max(inputs) > right:
        print("Input outside domain") #Error

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    unnorm_heights_exp = F.softplus(unnormalized_heights) + 1e-3

    if unnorm_heights_exp.shape[-1] == num_bins - 1:
        # Set boundary heights s.t. after normalization they are exactly 1.
        first_widths = 0.5 * widths[..., 0]
        last_widths = 0.5 * widths[..., -1]
        numerator = (
            0.5 * first_widths * unnorm_heights_exp[..., 0]
            + 0.5 * last_widths * unnorm_heights_exp[..., -1]
            + torch.sum(
                ((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2)
                * widths[..., 1:-1],
                dim=-1,
            )
        )
        constant = numerator / (1 - 0.5 * first_widths - 0.5 * last_widths)
        constant = constant[..., None]
        unnorm_heights_exp = torch.cat([constant, unnorm_heights_exp, constant], dim=-1)

    unnormalized_area = torch.sum(
        ((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2) * widths,
        dim=-1,
    )[..., None]
    heights = unnorm_heights_exp / unnormalized_area
    heights = min_bin_height + (1 - min_bin_height) * heights

    bin_left_cdf = torch.cumsum(
        ((heights[..., :-1] + heights[..., 1:]) / 2) * widths, dim=-1
    )
    bin_left_cdf[..., -1] = 1.0
    bin_left_cdf = F.pad(bin_left_cdf, pad=(1, 0), mode="constant", value=0.0)

    bin_locations = torch.cumsum(widths, dim=-1)
    bin_locations[..., -1] = 1.0
    bin_locations = F.pad(bin_locations, pad=(1, 0), mode="constant", value=0.0)

    if inverse:
        bin_idx = torchutils.searchsorted(bin_left_cdf, inputs)[..., None]
    else:
        bin_idx = torchutils.searchsorted(bin_locations, inputs)[..., None]

    #print(bin_locations)
    #print(bin_idx)

    input_bin_locations = bin_locations.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_left_cdf = bin_left_cdf.gather(-1, bin_idx)[..., 0]

    input_left_heights = heights.gather(-1, bin_idx)[..., 0]
    input_right_heights = heights.gather(-1, bin_idx + 1)[..., 0]

    a = 0.5 * (input_right_heights - input_left_heights) * input_bin_widths
    b = input_left_heights * input_bin_widths
    c = input_left_cdf

    if inverse:
        c_ = c - inputs
        alpha = (-b + torch.sqrt(b.pow(2) - 4 * a * c_)) / (2 * a)
        outputs = alpha * input_bin_widths + input_bin_locations
        outputs = torch.clamp(outputs, 0, 1)
        logabsdet = -torch.log(
            (alpha * (input_right_heights - input_left_heights) + input_left_heights)
        )
    else:
        alpha = (inputs - input_bin_locations) / input_bin_widths
        outputs = a * alpha.pow(2) + b * alpha + c
        outputs = torch.clamp(outputs, 0, 1)
        logabsdet = torch.log(
            (alpha * (input_right_heights - input_left_heights) + input_left_heights)
        )

    if inverse:
        outputs = outputs * (right - left) + left
    else:
        outputs = outputs * (top - bottom) + bottom

    return outputs, logabsdet
        
class QuadraticsSpline(nn.Module): #COMMENTS
    """
    Module used to apply a quadratic coupling layer (look for better terminology)
    ------------
    Forward mapping process with scaling and conditioning: (here @ is the Hadamar matrix product)

        z = (z1,z2) where size of z1 is (n)
        z2c = (z2,c) , concat
        params height and width are computed by passing z2c through the nn s
        x1 is computed through quadratic splines with previous params height and width
        x = (x1,z2)
                
    see for reference:  https://github.com/bayesiains/nflows
                        https://arxiv.org/pdf/1906.04032.pdf
                        https://arxiv.org/pdf/2106.05285.pdf
    """

    def __init__(self, input_dim: int, output_dim: int, n: int, K: int, s):
        """
        Arguments:
            - input_dim: int: dimension of the input or 'label'
            - output_dim: int: dimension of the data
            - n: int: number coordinates of initial vector that will be preserved through the identity transform
            - s: lambda x: Torch.tensor of size (data_dim - n + input_dim) -> Torch.tensor of size(n), point wise operation function,
                operating on the last dim of a Torch.tensor
                can be a trainable neural network
                Gives parameters of the quadratic spline

    
        """
        super().__init__()

        assert n<=output_dim

        self.n = n
        self.s = s
        self.K = K


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

            #Split tensor
            (z1,z2) = torch.split(z, (self.n, data_dim - self.n), dim = 1)

            z2c = torch.concat((z2,c), dim=1)
            y=self.s(z2c)
            scaling_vector = torch.reshape(y,[y.shape[0],self.n,2*self.K+1])
            (unnormalized_heights,unnormalized_widths)=torch.split(scaling_vector, self.K+1,2)
            #print(unnormalized_heights)
            #print(unnormalized_widths)
            #print(z1)

            x1, log_det=quadratic_spline(z1,unnormalized_widths, unnormalized_heights, reverse)
            x2 = z2

            x = torch.concat((x2,x1), dim=1)

            return x, log_det.sum(1)
        else:

            #Split tensor
            #print(z.shape)
            (x1,x2) = torch.split(z, (self.n, data_dim - self.n), dim = 1)

            x2c = torch.concat((x2,c), dim=1)
            y=self.s(x2c)
            #print(y.shape)
            scaling_vector = torch.reshape(y,[y.shape[0], self.n,2*self.K+1])
            (heights,widths)=torch.split(scaling_vector, self.K+1,2)
            #heights,indx=torch.sort(heights,dim=-1)
            #widths=F.normalize(widths,p=1,dim=-1)
            heights=torch.sigmoid(heights)
            widths=torch.sigmoid(widths)
            print(heights,widths)
            #print(x1)
            
            z1, log_det = quadratic_spline(x1,widths, heights, reverse)
            z2 = x2

            z = torch.concat((z2,z1), dim=1)
            loss=log_det.sum(1)
            #print(loss)

            return z, loss