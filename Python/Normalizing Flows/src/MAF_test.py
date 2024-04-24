import torch
from flows import ConditionaMAF
from tests import *
from datasets import TestSet1
from misc_transforms import InvertibleMapping

torch.set_default_dtype(torch.float64)

flow = torch.load('Normalizing Flows/models/MAF.pt')
#flow = ConditionaMAF(3, 1, 3)

is_cond_flow_invertible(flow, verbose=True)
is_cond_flow_invertible(flow, verbose=True, reversed=True)

sample_flow_discreet(flow, 100)

sample_flow_1D(flow)