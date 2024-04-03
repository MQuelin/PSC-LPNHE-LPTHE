from pathlib import Path
import torch
import matplotlib.pyplot as plt
from datasets import ZeeDataset

# flow = torch.load(Path(__file__) / Path('../../models/CNF2_test_3.pt')).to('cuda')

# c = torch.Tensor([[-1+k/125,1.0,1.0] for k in range(250)]).to('cuda').to(torch.float64)

# samples, log_jac = flow.sample(c)

# plt.scatter(c[::,0].to('cpu'), samples[::,0].detach().to('cpu'))
# plt.show()

# data = ZeeDataset()

# e = []
# e_truth = []

# for k in range(100):
#     sample = data[k]
#     e_truth.append(sample['input'][0])
#     e.append(sample['output'][0])

# plt.scatter(e_truth,e)
# plt.show()

a = torch.rand([10,3])
b = torch.nn.Parameter(torch.ones(1))

print(a)
print(a[::,1])

a[::,0] = torch.rand(10)

print(a)