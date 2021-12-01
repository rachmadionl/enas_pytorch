import sys
import os
import numpy as np


path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(path)
os.chdir(path)
sys.path.append(path)
from child import Child


def test_train():
    num_layer = 6
    dataset = 'cifar10'
    if dataset == 'cifar10':
        in_channels = 3
    elif dataset == 'mnist':
        in_channels = 1
    child = Child(in_channels=in_channels, num_layers=num_layer, dataset=dataset)
    num_epoch = 30
    sample_arch = []
    for i in range(num_layer):
        op_config = list(np.random.randint(0, 4, size=1))
        skip_config = list(np.random.randint(0, 2, size=i))
        if i == 0:
            sample_arch.append([op_config, []])
        else:
            sample_arch.append([op_config, skip_config])

    child.fit(sample_arch, num_epoch=num_epoch)
    child.valid(sample_arch)


if __name__ == '__main__':
    test_train()
