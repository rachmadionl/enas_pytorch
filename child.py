from typing import List

import torch
import torch.nn as nn

from shared_cnn import NASConvModel


class Child:
    def __init__(self,
                 in_channels: int = 3,
                 class_num: int = 10,
                 num_layers: int = 6,
                 feature_size: int = 32,
                 batch_size: int = 32,
                 lr: float = 0.05,
                 lr_cos_min: float = 0.001,
                 lr_cos_max: float = 2,
                 l2_reg: float = 1e-4,
                 dataset='cifar10'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels
        self.class_num = class_num
        self.num_layers = num_layers
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.lr = lr

        self.net = NASConvModel(in_channels, class_num, num_layers, feature_size)
        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD([{'params': self.net.parameters(), 'initial_lr': lr}],
                                     lr=lr, weight_decay=l2_reg, momentum=0.9, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, lr_cos_max, eta_min=lr_cos_min)
        self.dataset = dataset

    def fit(self, sample_arch: List[int], num_epoch: int, train_loader):
        print('lr=', self.scheduler.get_last_lr())
        # trainloader = data_loader(self.batch_size, dataset=self.dataset)
        for epoch in range(num_epoch):
            total_loss = 0.0
            for step, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optim.zero_grad()

                outputs = self.net(inputs, sample_arch)
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optim.step()

                total_loss += loss.item()
                if step % 100 == 99:
                    print(f'step: {step + 1:4d}\tloss: {total_loss / 100:.3f}')
                    total_loss = 0.0
            self.scheduler.step()

    def valid_rl(self, sample_arch: List[int], valid_loader):
        # validloader = data_loader(self.batch_size, mode='valid', dataset=self.dataset)
        data = next(iter(valid_loader))
        images, labels = data[0].to(self.device), data[1].to(self.device)

        outputs = self.net(images, sample_arch)

        _, idx = torch.topk(outputs, 1)
        idx = idx.reshape((-1))
        acc = (idx == labels).float().sum()

        acc /= self.batch_size
        return acc

    def valid(self, sample_arch: List[int], valid_loader):
        # validloader = data_loader(self.batch_size, mode='valid', dataset=self.dataset)
        total_acc = 0
        for data in valid_loader:
            images, labels = data[0].to(self.device), data[1].to(self.device)

            outputs = self.net(images, sample_arch)

            _, idx = torch.topk(outputs, 1)
            idx = idx.reshape((-1))
            total_acc += (idx == labels).float().sum()

        total_acc /= (len(valid_loader) * self.batch_size)

        return total_acc
