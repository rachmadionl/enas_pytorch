import torch
import torch.nn as nn
from typing import List

from shared_cnn import NASConvModel
from loader import data_loader


class Child:
    def __init__(self,
                 in_channels: int = 3,
                 class_num: int = 10,
                 num_layers: int = 6,
                 feature_size: int = 64,
                 batch_size: int = 32,
                 lr: float = 8e-3,
                 dataset='cifar10'):
        self.in_channels = in_channels
        self.class_num = class_num
        self.num_layers = num_layers
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = NASConvModel(in_channels, class_num, num_layers, feature_size)
        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.net.parameters(), self.lr)

        self.dataset = dataset

    def fit(self, sample_arch: List[int], num_epoch: int):
        trainloader = data_loader(self.batch_size, dataset=self.dataset)
        self.net.train()
        for epoch in range(num_epoch):
            total_loss = 0.0
            for step, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optim.zero_grad()

                outputs = self.net(inputs, sample_arch)
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optim.step()

                total_loss += loss.item()
                if step % 100 == 99:
                    print(f'epoch: {epoch + 1:2d}\tstep: {step + 1:4d}\tloss: {total_loss / 100:.3f}')
                    total_loss = 0.0

    def valid(self, sample_arch: List[int]):
        validloader = data_loader(self.batch_size, mode='valid', dataset=self.dataset)
        total_acc = 0
        self.net.eval()
        for data in validloader:
            images, labels = data[0].to(self.device), data[1].to(self.device)

            outputs = self.net(images, sample_arch)

            _, idx = torch.topk(outputs, 1)
            idx = idx.reshape((-1))
            total_acc += (idx == labels).float().sum()

        total_acc /= (len(validloader) * self.batch_size)
        print(f'Total Accruacy is {total_acc:.3f}')
        return total_acc
