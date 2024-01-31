import torch
from tqdm import tqdm

from pathlib import Path

from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal

from flow_utils import gaussian_log_pdf

class Trainer:

    """
    Tainer used to train a model to transtion from a normal distribution to a target distribution represented by a set of samples.
    """

    def __init__(self, flow, optimizer, dataloader, data_dim, device) -> None:
        self.flow = flow
        self.flow.to(device)
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.data_dim = data_dim
        self.device = device
        


    def train(self, nb_epochs):
        training_loss = []
        for epoch in tqdm(range(nb_epochs)):

            for batch, sample in enumerate(self.dataloader):
                # Propagate the samples backwards through the flow
                x = sample['output'].to(self.device)
                z, log_jac_det = self.flow(x, reverse=True)

                # Evaluate loss
                loss =  (0.5*torch.sum(z**2, 1) - log_jac_det).mean() / self.data_dim

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                training_loss.append(loss.item())

        return training_loss
    
    def save_at(self, save_path = '', save_name = 'NFModel.pt') :
        absolute_path = Path(__file__).parent
        save_path = absolute_path/save_path/save_name

        print(f'Saving model at {save_path}')
        torch.save(self.flow, save_path)

class ConditionalTrainer:

    """
    Tainer used to train a model to transtion from a normal distribution to a target distribution represented by a set of samples, 
    where each sample is conditionned by a vector c
    """

    def __init__(self, flow, optimizer, dataloader, input_dim, output_dim, epsilon, device) -> None:
        self.flow = flow
        self.flow.to(device)
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.device = device
        


    def train(self, nb_epochs):
        training_loss = []
        for epoch in tqdm(range(nb_epochs)):

            for batch, sample in enumerate(self.dataloader):
                # Propagate the samples backwards through the flow
                c = sample['input'].to(self.device)
                x = sample['output'].to(self.device)
                z, log_jac_det = self.flow(c, x, reverse=True)

                c_estimate, dummy_variable = torch.split(z, (self.input_dim, self.output_dim-self.input_dim), dim = 1)

                # Evaluate loss
                loss =  (0.5*torch.sum(dummy_variable**2, 1) + 0*5/self.epsilon*torch.sum((c_estimate-c)**2, 1) - log_jac_det).mean() / self.output_dim

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                training_loss.append(loss.item())

        return training_loss
    
    def save_at(self, save_path = '', save_name = 'NFModel.pt') :
        absolute_path = Path(__file__).parent
        save_path = absolute_path/save_path/save_name

        print(f'Saving model at {save_path}')
        torch.save(self.flow, save_path)