import torch
from my_bijective_transforms import quadratic_spline
from my_flows import ConditionalQS

K=5
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"

# input=torch.randn(5,5)
# input=torch.sigmoid(input)
# width=torch.rand(5,5,K)
# height=torch.rand(5,5,K+1)
# height,idx=torch.sort(height)

# a = torch.arange(20).reshape(5, 2, 2)

# print(width,height)
# print(input)
# print(quadratic_spline(input,width, height))

batch_size = 100

flow = ConditionalQS(1, 3, 10, 5, device=device)

labels = torch.randn(10, flow.input_dim).to(device)
batch = torch.randn(10, flow.output_dim).to(device)

output,log_det=flow(labels,batch)
print(output,log_det)