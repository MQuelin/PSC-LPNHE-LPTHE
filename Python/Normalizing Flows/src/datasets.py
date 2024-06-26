import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path
from math import pi
import matplotlib.pyplot as plt

#Sanitize Inputs when loading instead of simply commenting out

class ZeeDataset(Dataset):

    def __init__(self, relative_path='../data/data_dict.pkl') -> None:
        """
        Loads the Zee colision data pickle located at the specified relative_path
        """
        
        absolute_path = Path(__file__).parent
        relative_path = relative_path
        open_path = absolute_path / relative_path

        with open(open_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        print('Loading dataset ...')

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
        # x.append(data_dict['weta2']) causes nan to appear when normalized, reason unknown as of now, commented out
        x.append(data_dict['f1'])
        x.append(data_dict['f3'])
        
        self.inputs = torch.tensor([z1,z2,z3]).transpose(0,1)
        self.outputs = torch.tensor([xi for xi in x]).transpose(0,1)

        #Normalization
        self._in_mean = torch.mean(self.inputs, dim=0, keepdim = True)
        self._in_std = torch.std(self.inputs, dim=0, keepdim = True)
        self._out_mean = torch.mean(self.outputs, dim=0, keepdim = True)
        self._out_std = torch.std(self.outputs, dim=0, keepdim = True)

        self.inputs = (self.inputs - self._in_mean) / self._in_std
        self.outputs = (self.outputs - self._out_mean) / self._out_std
        
        print('Dataset Loaded !')

    def __len__(self):
        return self.outputs.shape[0] - 1

    def __getitem__(self, idx) :
        sample_input = self.inputs[idx]
        sample_output = self.outputs[idx]
        sample = {'input': sample_input, 'output': sample_output}
        return sample

class TestSet1(Dataset):
    def __init__(self, n, plot_data = False) -> None:
        super().__init__()

        radius = 5.0

        self.__len = n
        self.__labels = torch.Tensor([2*pi*i/n - pi/2 for i in range(n)])
        self.__data = radius*torch.cos(torch.Tensor([[2*pi*i/n - pi/2, 2*pi*i/n - pi, 0] for i in range(n)])) + torch.randn(size=(n,3))

        if plot_data:
            plt.scatter(self.__data[::1000,0].numpy(), self.__data[::1000,1].numpy())
            plt.show()

    def __len__(self):
        return self.__len

    def __getitem__(self, idx) :
        sample_input = self.__labels[idx]
        sample_output = self.__data[idx]
        sample = {'input': sample_input, 'output': sample_output}
        return sample

class TestSet2(Dataset):
    def __init__(self, n, plot_data = False) -> None:
        super().__init__()

        radius = 5.0

        self.__len = n
        self.__labels = torch.Tensor([int(2*i/n)*pi for i in range(n)])
        self.__data = radius*torch.cos(torch.Tensor([[int(2*i/n)*pi, int(2*i/n)*pi - pi/2, 0] for i in range(n)])) + torch.randn(size=(n,3))

        if plot_data:
            plt.scatter(self.__data[::1000,0].numpy(), self.__data[::1000,1].numpy())
            plt.show()

    def __len__(self):
        return self.__len

    def __getitem__(self, idx) :
        sample_input = self.__labels[idx]
        sample_output = self.__data[idx]
        sample = {'input': sample_input, 'output': sample_output}
        return sample

class TestSetGauss(Dataset):
    def __init__(self, n, donnee, plot_data = False):
        super().__init__()

        self.__len = n
        self.__labels = torch.Tensor([1 for i in range(n)])
        self.__data = torch.Tensor(plot_data)

        if plot_data:
            plt.scatter(self.__data[::1000,0].numpy(), self.__data[::1000,1].numpy())
            plt.show()

    def __len__(self):
        return self.__len

    def __getitem__(self, idx) :
        sample_input = self.__labels[idx]
        sample_output = self.__data[idx]
        sample = {'input': sample_input, 'output': sample_output}
        return sample