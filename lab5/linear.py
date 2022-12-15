import torch
import numpy
from torch import nn

dimension_layer = nn.Linear(1, 12)
input = torch.randn(1,7)
input = input.view(1,7)
print(input.shape)
print(input)
input = input.transpose(0, 1)
print(input.shape)
print(input)

output = dimension_layer(input)
print(output.shape)
print(output)

output = output.transpose(0, 1)
print(output.shape)
print(output)


