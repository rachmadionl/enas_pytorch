import torch

from controller import Controller
from child import Child
from loader import data_loader


def train():
    # Hyperparams
    batch_size = 64
    child_num_layers = 6
    epochs = 600
    dataset = 'mnist'
    if dataset == 'mnist':
        in_channels = 1
    elif dataset == 'cifar10':
        in_channels = 3
    train_step_num = 50
    sample_size = 10
    train_controller_every_n = 1

    train_loader, valid_loader = data_loader(batch_size=batch_size, dataset=dataset)
    # Create child
    child = Child(in_channels=in_channels, batch_size=batch_size, num_layers=child_num_layers, dataset=dataset)
    ctrlr = Controller(child_num_layers=child_num_layers, train_step_num=train_step_num, sample_size=sample_size)
    print('layer num of a child:', len(list(child.net.model)))
    child_acc = []
    ctrlr_acc = []

    for epoch in range(epochs):
        # 1. Sample an arch
        arc_sample = ctrlr.sample_arc()

        # 2. Train a sampled arch
        print('---------- Training a Child model ----------')
        child.fit(arc_sample, num_epoch=1, train_loader=train_loader)
        print('------------ End training child ------------\n')

        # 3.  Valid a child model
        print('---------- Validate a child model ----------')
        acc = child.valid(arc_sample, valid_loader=valid_loader)
        child_acc.append(acc)
        print(f'epoch: {epoch:2d}\tchild accuracy: {acc:.5f}')
        print('-------- End validate a child model --------\n')

        # 4. Train the controller
        if (epoch + 1) % train_controller_every_n == 0:
            print('---------- Train controller ----------')
            ctrlr.fit(child, valid_loader=valid_loader)
            print('-------- End training controller --------\n')

        # 5. Valid the controller
        print('---------- Validate controller ----------')
        acc_ctrlr = ctrlr.valid(child, arc_num=sample_size, valid_loader=valid_loader)
        acc_avg = torch.mean(torch.tensor(acc_ctrlr)).cpu().numpy()
        ctrlr_acc.append(acc_avg)
        print(f'epoch: {epoch:2d}\tcontroller avg acc: {acc_avg:.5f}')
        print('---------- End validate controller ----------\n')

    # get best arch
    best_acc, best_arc = ctrlr.get_best_arc(child, arc_num=sample_size, valid_loader=valid_loader)
    print(f'BEST ARC: {best_arc}\tBEST ACCURACY: {best_acc:.5f}')


if __name__ == '__main__':
    train()
