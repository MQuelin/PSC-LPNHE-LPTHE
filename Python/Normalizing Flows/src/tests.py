from pathlib import Path
import torch
import numpy as np

from scipy.stats import multivariate_normal

flow = torch.load(Path(__file__) / Path('../../models/ICNF_5.pt')).to('cpu')

test_batch = torch.Tensor([[1.],[2.]])

gaussian = multivariate_normal(cov=np.eye(28*28))
dummy_variables = torch.tensor(gaussian.rvs(size = 2)).to(torch.float32)

output, log_jac = flow(test_batch, dummy_variables, reverse = False)

reversed_output, log_jac_reversed = flow(test_batch, output, reverse = True)

print(f'Conditions: {test_batch}\nModel input: {dummy_variables}\nModel output: {output}\nReversed output: {reversed_output}\n\n')
print(f'log_jac = {torch.exp(log_jac)}\nlog_jac_reversered = {torch.exp(log_jac_reversed)}')

for k in range(728):
    print(dummy_variables[0,k]-reversed_output[0,k])