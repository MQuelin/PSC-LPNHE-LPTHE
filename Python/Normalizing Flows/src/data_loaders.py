import pickle
import torch
from sklearn.neighbors import KernelDensity
from scipy.stats import uniform
from pathlib import Path

class ZeeDataset:

    def __init__(self, relative_path: str) -> None:
        """
        Loads the Zee colision data pickle located at the specified relative_path
        """
        
        absolute_path = Path(__file__).parent
        relative_path = relative_path
        open_path = absolute_path / relative_path

        with open(open_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        z1 = data_dict['e_truth']
        z2 = data_dict['eta_truth']
        z3 = data_dict['phi_truth']
        z4 = data_dict['px_truth']
        z5 = data_dict['py_truth']
        z6 = data_dict['pz_truth']

        x1 = data_dict['e']
        x2 = data_dict['et']
        x3 = data_dict['eta']
        x4 = data_dict['phi']
        x5 = data_dict['reta']
        x6 = data_dict['rphi']
        
        
        inputs = torch.tensor([z1,z2,z3,z4,z5,z6]).transpose(0,1)
        self.KDE_of_inputs = KernelDensity(bandwidth="silverman")
        self.KDE_of_inputs.fit(inputs)
        
        outputs = torch.tensor([x1,x2,x3,x4,x5,x6]).transpose(0,1)
        self.KDE_of_outputs = KernelDensity(bandwidth="silverman")
        self.KDE_of_outputs.fit(outputs)

class TempUniform:
    """
    Load a simple uniform probability over [0,1]^N where N is data_dim

    Temporary, meant to be used for testing purposes
    """
    def __init__(self, data_dim) -> None:
        data = torch.tensor(uniform.rvs(size=(10000,data_dim)))
        self.KDE_of_outputs = KernelDensity(bandwidth=0.01)
        self.KDE_of_outputs.fit(data)
