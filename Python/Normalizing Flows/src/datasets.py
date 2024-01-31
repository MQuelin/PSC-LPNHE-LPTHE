import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.functional import normalize
from pathlib import Path

class ZeeDataset(Dataset):

    def __init__(self, relative_path: str) -> None:
        """
        Loads the Zee colision data pickle located at the specified relative_path
        """
        
        absolute_path = Path(__file__).parent
        relative_path = relative_path
        open_path = absolute_path / relative_path

        with open(open_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        print('Now loading dataset ...')

        z1 = data_dict['e_truth']
        z2 = data_dict['eta_truth']
        z3 = data_dict['phi_truth']

        x1 = data_dict['e']
        x2 = data_dict['et']
        x3 = data_dict['eta']
        x4 = data_dict['phi']
        x5 = data_dict['reta']
        x6 = data_dict['rphi']
        
        self.inputs = normalize(torch.tensor([z1,z2,z3]).transpose(0,1), dim= 0)
        self.outputs = normalize(torch.tensor([x1,x2,x3,x4,x5,x6]).transpose(0,1), dim = 0)
        print('Dataset Loaded !')

    def __len__(self):
        return self.inputs.shape[0] - 1

    def __getitem__(self, idx) :
        sample_input = self.inputs[idx]
        sample_output = self.outputs[idx]
        sample = {'input': sample_input, 'output': sample_output}
        return sample