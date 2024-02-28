from models import Mnist_Model
import torchvision
import torch
import numpy as np
from torchvision import datasets, transforms
from models import Mnist_Model

if __name__ == '__main__':
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    data_train = datasets.MNIST(root="../data/mnist",
                                transform=apply_transform,
                                train=True,
                                download=False)

    data_test = datasets.MNIST(root="../data/mnist",
                               transform=apply_transform,
                               train=False,
                               download=True)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=64,
                                                    shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=64,
                                                   shuffle=True)
    model = Mnist_Model()
    for batch_idx, (x, target) in enumerate(data_loader_train):
        print(x.shape)
        print(model(x))
