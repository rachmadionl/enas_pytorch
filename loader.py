import torch
import torchvision
import torchvision.transforms as transforms

# mean and std of CIFAR10 dataset
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)


def loader_cifar10(batch_size: int = 32):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Pad(2),
        transforms.RandomResizedCrop(32)
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    validset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=False, transform=valid_transform)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)

    return trainloader, validloader


def loader_mnist(batch_size: int = 32):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
        transforms.Pad(2),
        transforms.RandomResizedCrop(28)
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
        ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    validset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=valid_transform)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)

    return trainloader, validloader


def data_loader(batch_size: int = 32, dataset: str = 'cifar10'):
    if dataset == 'cifar10':
        return loader_cifar10(batch_size)
    elif dataset == 'mnist':
        return loader_mnist(batch_size)
