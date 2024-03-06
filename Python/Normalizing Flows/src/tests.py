import pickle
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# absolute_path = Path(__file__).parent
# relative_path = '../data/100k3.pkl'
# open_path = absolute_path / relative_path

# with open(open_path, 'rb') as f:
#     data_dict = pickle.load(f)
#     print('Now loading dataset ...')

# print('Done')

# absolute_path = Path(__file__).parent
# relative_path = '../models/CNF_24layers_100k3Zee_noRings_1e-2_140224.pt'
# open_path = absolute_path / relative_path

# flow = torch.load(open_path)

# entry = torch.Tensor([[1.,1.,-0.5+k/50.] for k in range(101)])

# result, _jacob = flow.sample(entry)
# entry = entry.transpose(0,1).to('cpu').detach().numpy()
# result = result.transpose(0,1).to('cpu').detach().numpy()
# print(entry.shape)
# print(result.shape)

# n, bins, patches = plt.hist(result[3,:], 50, density=True, facecolor='g', alpha=0.75)
# plt.show()

# n, bins, patches = plt.hist(data_dict['phi'][0:100], 50, density=True, facecolor='g', alpha=0.75)
# plt.show()