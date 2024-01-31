import pickle
from pathlib import Path

absolute_path = Path(__file__).parent
relative_path = '../data/data_dict.pkl'
open_path = absolute_path / relative_path

with open(open_path, 'rb') as f:
    data_dict = pickle.load(f)
    print('Now loading dataset ...')

print('Done')

print(data_dict.keys())