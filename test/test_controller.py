import sys
import os
import numpy as np

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(path)
os.chdir(path)
sys.path.append(path)
from controller import Controller
from child import Child
from loader import data_loader

def test_controller():
    dataset='mnist'
    if dataset =='mnist':
        in_channels = 1
    elif dataset =='cifar10':
        in_channels = 3
    _, valid_loader = data_loader(dataset=dataset)
    num_layers = 6
    ctrlr = Controller(child_num_layers=num_layers, train_step_num=20, sample_size=10)
    child = Child(in_channels=in_channels, num_layers=num_layers, dataset=dataset)
    num_epoch = 10
    sample_arch = []
    for i in range(num_layers):
        op_config = list(np.random.randint(0, 4, size=1))
        skip_config = list(np.random.randint(0, 2, size=i))
        if i == 0:
            sample_arch.append(op_config)
            sample_arch.append(skip_config)
        else:
            sample_arch.append(op_config)
            sample_arch.append(skip_config)

    print('----- controller training -----')
    ctrlr.fit(child)
    print('----- controller valid -----')
    ctrlr.valid(child, arc_num=10, valid_loader=valid_loader)
    print('----- controller get_best_arc -----')
    ctrlr.get_best_arc(child, arc_num=10, valid_loader=valid_loader)

if __name__ == '__main__':
    test_controller()
