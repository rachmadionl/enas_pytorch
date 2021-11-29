import sys
import os
import pytest
import torch

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(path)
os.chdir(path)
sys.path.append(path)
from shared_cnn import NASConv, ConvCell


def test_convcell():
    inputs = torch.rand([2, 3, 32, 32])
    conv = ConvCell("cv3", feature_size=3)

    output = conv(inputs)

    assert output.size() == torch.Size([2, 3, 32, 32])

def test_nasconv():
    N = 2; C = 3; H = 32; W = 32
    inputs = torch.rand([N, C, H, W])
    conv_layer = NASConv(C)
    num_layer = 4
    prev_layers = []
    for i in range(num_layer):
        prev_layers.append(torch.rand(inputs.size()))
    op_config = [0]
    skip_config = [0, 1, 1, 0]
    output = conv_layer(prev_layers, [op_config, skip_config])

    assert output.size() == torch.Size([2, 3, 32, 32])