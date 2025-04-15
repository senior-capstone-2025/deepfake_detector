import torch
from torch import nn
import torch.nn.functional as F

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        bias = bias.view(1, bias.shape[0], *rest_dim)
        input = input + bias
    
    return F.leaky_relu(input, negative_slope) * scale