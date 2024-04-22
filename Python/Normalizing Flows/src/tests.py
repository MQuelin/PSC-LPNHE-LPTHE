import torch

"""
This module contains functions used to ensure that a normalizing flow works properly
for example, check that sampling works or that the network is invertible

If you wish to do personnal tests:
    - create a file in Python/Normalizing Flows/src named personnal_test.py
    -> this file is in the .gitignore and will therefore not be modified so that you can keep personnal tests in that file
"""


def is_flow_invertible(flow, eps=1e-6, verbose=False):
    batch = torch.randn(10, flow.output_dim)
    output, det = flow(batch, reverse = False)
    output_inv, det_inv = flow(output, reverse = True)

    max_err = torch.max(torch.abs(batch-output_inv))
    if max_err<eps:
        if verbose:
            print(f'flow is invertible, with max error encountered {max_err}\n\n')
        return(True)
    else:
        if verbose:
            print(f'flow is not invertible, with max error encountered {max_err}\n')
            print(f'det = {det}\ndet_inv = {det_inv}\n\n')
        return(False)