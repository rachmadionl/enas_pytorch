from typing import List

import torch
import torch.nn as nn


DEBUG = True


def conv3x3(in_channels: int, out_channels: int):
    pad = int((3 - 1) / 2)
    pad = (pad, pad)
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=pad)


def conv5x5(in_channels: int, out_channels: int):
    pad = int((5 - 1) / 2)
    pad = (pad, pad)
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=pad)


def maxpool3x3():
    pad = int((3 - 1) / 2)
    pad = (pad, pad)
    return nn.MaxPool2d(kernel_size=3, padding=pad)


def maxpool5x5():
    pad = int((5 - 1) / 2)
    pad = (pad, pad)
    return nn.MaxPool2d(kernel_size=5, padding=pad)


class ConvCell(nn.Module):
    """ An Operation used inside a NAS Layer.
        This cell consists of Conv2D-ReLU-BatchNorm-Kernel.
        A kernel is an operation between: \n
            1. "cv3": conv3x3      (index 0)
            2. "cv5": conv5x5      (index 1)
            3. "mp3": maxpool3x3   (index 2)
            4. "mp5": maxpool5x5   (index 3)

        Kernel gets picked randomly by a Controller.

        Args:
                - op (str): Name of the Operation.
                - feature_size (int): Size of the features.

        Return:
                - A Convolutional cell.
    """
    def __init__(self, op: str, feature_size: int):
        super().__init__()
        kernel = {
            "cv3": conv3x3(feature_size, feature_size),
            "cv5": conv5x5(feature_size, feature_size),
            "mp3": maxpool3x3(),
            "mp5": maxpool5x5()
            }

        cell = [
            nn.Conv2d(feature_size, feature_size, kernel_size=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(),
            kernel[op]
        ]

        self.cell = nn.ModuleList(cell)

    def forward(self, x):
        for layer in self.cell:
            x = layer(x)
        return x


class NASConv(nn.Module):
    """ A NAS Layer.

        Args:
             - feature_size (int): Size of the features.
    """
    def __init__(self, feature_size: int):
        super(NASConv, self).__init__()
        self.feature_size = feature_size
        self.layer_list = self._build_nas_layer()
        self.batch_norm = nn.BatchNorm2d(self.feature_size)

    def _build_nas_layer(self):
        layer_list = [
            ConvCell("cv3", self.feature_size),
            ConvCell("cv5", self.feature_size),
            ConvCell("mp3", self.feature_size),
            ConvCell("mp5", self.feature_size)
        ]
        layer_list = nn.ModuleList(layer_list)

        return layer_list

    def _skip_connection(self, x: torch.Tensor, prev_layers: List[torch.Tensor], skip_config: List[int]):
        y = []
        # offset by 1 since the first layer of the model is not a NAS Layer
        offset = 1
        num_layers = len(prev_layers) - offset
        for i in range(num_layers):
            if skip_config[i]:
                y.append(prev_layers[i + offset])
        if len(y):
            y = torch.sum(torch.stack(y), dim=0)
            x = torch.sum(torch.stack([x, y]), dim=0)
            
        return x

    def forward(self, prev_layers: List[torch.Tensor], layer_config: List[List[int]]):
        # input
        x = prev_layers[-1]
        op_config, skip_config = layer_config[0], layer_config[1]
        x = self.layer_list[op_config[0]](x)
        if len(prev_layers) > 0:  # there's no prev_layers for the 1st layer
            x = self._skip_connection(x, prev_layers, skip_config)

        if op_config[0] == 0 or op_config[0] == 1:  # if op is conv
            x = self.batch_norm(x)

        return x
