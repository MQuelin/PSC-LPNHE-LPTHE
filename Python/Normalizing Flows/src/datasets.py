import pickle
import torch
from torch.utils.data import Dataset, TensorDataset
from torch.nn.functional import normalize
from pathlib import Path
from numpy import array as np_array
from numpy import float32 as np_float32

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

        x= []
        x.append(data_dict['e'])
        x.append(data_dict['et'])
        x.append(data_dict['eta'])
        x.append(data_dict['phi'])
        x.append(data_dict['reta'])
        x.append(data_dict['rphi'])
        x.append(data_dict['rhad'])
        x.append(data_dict['eratio'])
        # x.append(data_dict['weta2']) causes nan to appear when normalized, reason unknown as of know, commented out
        x.append(data_dict['f1'])
        x.append(data_dict['f3'])
        
        self.inputs = torch.tensor([z1,z2,z3]).transpose(0,1)
        self.outputs = torch.tensor([xi for xi in x]).transpose(0,1)

        #Normalization
        in_mean = torch.mean(self.inputs, dim=0, keepdim = True)
        in_std = torch.std(self.inputs, dim=0, keepdim = True)
        out_mean = torch.mean(self.outputs, dim=0, keepdim = True)
        out_std = torch.std(self.outputs, dim=0, keepdim = True)

        self.inputs = (self.inputs - in_mean) / in_std
        self.outputs = (self.outputs - out_mean) / out_std
        
        print('Dataset Loaded !')

    def __len__(self):
        return self.inputs.shape[0] - 1

    def __getitem__(self, idx) :
        sample_input = self.inputs[idx]
        sample_output = self.outputs[idx]
        sample = {'input': sample_input, 'output': sample_output}
        return sample