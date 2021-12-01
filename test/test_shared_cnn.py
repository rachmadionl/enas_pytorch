import sys
import os
import pytest
import torch
import numpy as np


path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(path)
os.chdir(path)
sys.path.append(path)
from shared_cnn import NASConv, ConvCell, NASConvModel


@pytest.mark.parametrize("op", ["cv3", "cv5", "mp3", "mp5"])
def test_convcell(op):
    inputs = torch.rand([2, 3, 32, 32])
    conv = ConvCell(op, feature_size=3)

    output = conv(inputs)

    assert output.size() == torch.Size([2, 3, 32, 32])


@pytest.mark.parametrize("op", [0, 1, 2, 3])
def test_nasconv(op):
    N = 2; C = 3; H = 32; W = 32
    inputs = torch.rand([N, C, H, W])
    conv_layer = NASConv(C)
    num_layer = 4
    prev_layers = []
    for i in range(num_layer):
        prev_layers.append(torch.rand(inputs.size()))
    op_config = [op]
    skip_config = [0, 1, 1, 0]
    output = conv_layer(prev_layers, [op_config, skip_config])

    assert output.size() == torch.Size([2, 3, 32, 32])


def test_model():
    N = 2; C = 3; H = 32; W = 32
    inputs = torch.randn([N, C, H, W])
    num_layer = 6
    net = NASConvModel(in_channels=3, class_num=10, num_layer=num_layer, feature_size=64)
    sample_arch = []
    for i in range(num_layer):
        op_config = list(np.random.randint(0, 4, size=1))
        skip_config = list(np.random.randint(0, 2, size=i))
        if i == 0:
            sample_arch.append([op_config, []])
        else:
            sample_arch.append([op_config, skip_config])

    outputs = net(inputs, sample_arch)

    assert outputs.size() == torch.Size([2, 10])