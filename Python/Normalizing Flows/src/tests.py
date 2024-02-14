# import pickle
# from pathlib import Path
# import torch

# absolute_path = Path(__file__).parent
# relative_path = '../data/data_dict.pkl'
# open_path = absolute_path / relative_path

# with open(open_path, 'rb') as f:
#     data_dict = pickle.load(f)
#     print('Now loading dataset ...')

# print('Done')

# path = Path(__file__).parent.parent / 'models' / 'ConditionalNF_12layers_10kZee_noRings_310124.pt'

# flow = torch.load(path)

# c = torch.Tensor([[-0.4041, -0.8285,  0.0926],[-0.4041, -0.8285,  0.0926]])

# print(flow.sample(c))

if (1 >= 0.5) and (1 <= 1.5):
    print('True')