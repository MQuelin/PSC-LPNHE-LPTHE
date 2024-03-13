from flows import ImageConditionalNF

import torch


import matplotlib.pyplot as plt

from pathlib import Path

flow = torch.load(Path(__file__) / Path('../../models/ICNF_5.pt'))

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    X = torch.squeeze(flow.sample(torch.Tensor([[float(0)] for k in range(2)])))
    figure.add_subplot(rows, cols, i)
    plt.title(str(0))
    plt.axis("off")
    plt.imshow(X.to('cpu').detach()[0], cmap="gray")
plt.show()