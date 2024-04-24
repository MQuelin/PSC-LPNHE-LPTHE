import torch
import matplotlib.pyplot as plt
from math import pi

"""
This module contains functions used to ensure that a normalizing flow works properly
for example, check that sampling works or that the network is invertible

If you wish to do personnal tests:
    - create a file in Python/Normalizing Flows/src named personnal_test.py
    -> this file is in the .gitignore and will therefore not be modified so that you can keep personnal tests in that file
"""


def is_flow_invertible(flow, eps=1e-6, verbose=False, reversed = False):
    batch = 10*torch.randn(10, flow.output_dim).to('cuda')
    output, det = flow(batch, reverse = reversed)
    output_inv, det_inv = flow(output, reverse = not reversed)

    max_err = torch.max(torch.abs(batch-output_inv))
    if max_err<eps:
        if verbose:
            print(f'reverse is {reversed}, flow is invertible with max error encountered {max_err}\n\n')
        return(True)
    else:
        if verbose:
            print(f'reverse is {reversed}, flow is not invertible with max error encountered {max_err}\n')
            print(f'det = {det}\ndet_inv = {det_inv}\n\n')
        return(False)

def is_cond_flow_invertible(flow, eps=1e-6, verbose=False, reversed = False):
    labels = torch.randn(10, flow.input_dim).to('cuda')
    # if flow.input_dim == 1:
    #     labels = labels.unsqueeze(dim=1)
    batch = 10*torch.randn(10, flow.output_dim).to('cuda')

    output, det = flow(c=labels, z=batch, reverse = False)
    output_inv, det_inv = flow(c=labels, z=output, reverse = True)

    max_err = torch.max(torch.abs(batch-output_inv))
    if max_err<eps:
        if verbose:
            print(f'reverse is {reversed}, flow is invertible with max error encountered {max_err}\n\n')
        return(True)
    else:
        if verbose:
            print(f'reverse is {reversed}, flow is not invertible with max error encountered {max_err}\n')
            print(f'det = {det}\ndet_inv = {det_inv}\n\n')
            print(batch[0])
            print(output_inv[0])
        return(False)

def sample_flow_1D(flow, n=100, start=0, end=pi, show_train_set=True):
    delta = end-start
    label_list = []
    for k in range(9):
        label_list = label_list + n*[start+delta*k/8]
    labels = torch.Tensor(label_list)

    samples, _log_jac = flow.sample(labels)
    samples = samples.cpu().detach()

    plt.figure(figsize=(7,7)) 
    for k in range(9):
        ax = plt.subplot(3, 3, k + 1)
        # ax.set_xlim([-8,8])
        # ax.set_ylim([-4,8])
        ax.scatter(samples[n*k:n*(k+1),0],samples[n*k:n*(k+1),1])
        if show_train_set:
            ax.scatter(torch.randn(size=(30,1)) + 5*torch.cos(labels[n*k:n*(k+1)]),
                        torch.randn(size=(30,1)) + 5*torch.sin(labels[n*k:n*(k+1)]))
            ax.set_xlim([-10,10])
            ax.set_ylim([-3,10])
        print(f'sample example: {samples[n*k-1]}')
    plt.show()

def sample_flow_discreet(flow, n=10, label_list=[0,pi], show_train_set=True):

    labels = []
    for label in label_list:
        labels = labels + n*[label]
    labels = torch.Tensor(labels)

    samples, _log_jac = flow.sample(labels)
    samples = samples.cpu().detach()

    plt.figure(figsize=(7,7)) 
    k=0
    for label in label_list:
        ax = plt.subplot(2, 1, k + 1)
        # ax.set_xlim([-8,8])
        # ax.set_ylim([-4,8])
        ax.scatter(samples[n*k:n*(k+1),0],samples[n*k:n*(k+1),1])
        if show_train_set:
            ax.scatter(torch.randn(size=(30,1)) + 5*torch.cos(labels[n*k:n*(k+1)]),
                        torch.randn(size=(30,1)) + 5*torch.sin(labels[n*k:n*(k+1)]))
            ax.set_xlim([-10,10])
            ax.set_ylim([-3,10])
        print(f'sample example: {samples[n*k-1]}')
        k = k+1
    plt.show()

def sample_flow(flow, n=100):
    samples = flow.sample(n)

    plt.scatter(samples[:,0], samples[:,1])
    plt.show()