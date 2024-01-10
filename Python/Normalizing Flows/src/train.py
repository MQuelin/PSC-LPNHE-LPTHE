import torch
from tqdm import tqdm

from pathlib import Path

from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal

from flow_utils import gaussian_log_pdf

class KernelBasedTrainer:
    """
    Tainer used to train a model to transtion from a given probality distribution to another.
    Both the input and target distribution have to be instances of sklearn.neighbors.KernelDensity
    """

    def __init__(self, flow, optimizer, training_input_density: KernelDensity, training_output_density: KernelDensity, device) -> None:
        self.device = device
        self.flow = flow
        self.flow.to(self.device)
        self.optimizer = optimizer
        self.training_input_density = training_input_density
        self.training_output_density = training_output_density
        


    def train_and_save(self, nb_epochs, batch_size, save_path = '', save_name = 'NFModel.pt'):
        training_loss = []
        
        for epoch in tqdm(range(nb_epochs)):
            # Generate new samples from the flow
            z0 = torch.tensor(self.training_input_density.sample(batch_size), dtype = torch.float32).to(self.device)
            zk, log_jacobian = self.flow(z0)

            # Evaluate the exact and approximated densities
            flow_log_density = torch.tensor(self.training_input_density.score_samples(z0.to('cpu'))) - log_jacobian.to('cpu') # This formula is a consequence of the change of variable done using the normalizing flow
            exact_log_density = torch.tensor(self.training_output_density.score_samples(zk.to('cpu').detach()))

            # Compute the loss
            reverse_kl_divergence = (flow_log_density - exact_log_density).mean()
            self.optimizer.zero_grad()
            loss = reverse_kl_divergence
            loss.backward()
            self.optimizer.step()

            training_loss.append(loss.item())
        
        absolute_path = Path(__file__).parent
        save_path = absolute_path/save_path/save_name

        # torch.save(self.flow, save_path)

        return training_loss

class NormTrainer:

    """
    Tainer used to train a model to transtion from a normal distribution to another.
    Both the target distribution has to be an instance of sklearn.neighbors.KernelDensity
    """

    def __init__(self, flow, optimizer, training_output_density: KernelDensity, data_dim, device) -> None:
        self.device = device
        self.flow = flow
        self.flow.to(self.device)
        self.optimizer = optimizer
        self.rv = multivariate_normal(mean=torch.zeros(data_dim), cov=torch.eye(data_dim), allow_singular=False)
        self.training_output_density = training_output_density
        


    def train_and_save(self, nb_epochs, batch_size, save_path = '', save_name = 'NFModel.pt'):
        training_loss = []
        
        for epoch in tqdm(range(nb_epochs)):
            # Generate new samples from the flow
            z0 = torch.tensor(self.rv.rvs(size=batch_size), dtype = torch.float32).to(self.device)
            zk, log_jacobian = self.flow(z0)

            # Evaluate the exact and approximated densities
            flow_log_density = gaussian_log_pdf(z0.to('cpu')).unsqueeze(1) + log_jacobian.to('cpu') # This formula is a consequence of the change of variable done using the normalizing flow
            exact_log_density = torch.tensor(self.training_output_density.score_samples(zk.to('cpu').detach())).unsqueeze(1)

            # Compute the loss
            reverse_kl_divergence = torch.mean(flow_log_density-exact_log_density)

            self.optimizer.zero_grad()
            loss = reverse_kl_divergence
            loss.backward()
            self.optimizer.step()

            training_loss.append(loss.item())
        
        absolute_path = Path(__file__).parent
        save_path = absolute_path/save_path/save_name

        # torch.save(self.flow, save_path)

        return training_loss