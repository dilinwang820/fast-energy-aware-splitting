from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os


base_data_dir = "../data/"
def get_data_loader(dataset='mnist', train_batch_size=100, test_batch_size=100, use_cuda=True):

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    if dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(base_data_dir, './data_mnist'), train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=train_batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(base_data_dir, './data_mnist'), train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)

    elif dataset == 'fashion_mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(os.path.join(base_data_dir, './data_fashion_mnist'), train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])),
            batch_size=train_batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.joinn(base_data_dir, './data_fashion_mnist'), train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)

    elif dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(base_data_dir, './data.cifar10'), train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Pad(4),
                               transforms.RandomCrop(32),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=train_batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(base_data_dir, './data.cifar10'), train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)

    elif dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(os.path.join(base_data_dir, './data.cifar100'), train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Pad(4),
                               transforms.RandomCrop(32),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=train_batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(os.path.join(base_data_dir, './data.cifar100'), train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)
     
    return train_loader, test_loader
