import torch
import torch.nn as nn

class OrthLinear(nn.Linear):
    def forward(self, input):
        return super.forward(input.to(self.weight.dtype))