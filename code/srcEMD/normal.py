import torchvision
from torch import nn
import torch
import numpy as np
from torchvision import datasets, transforms
from models import Mnist_Model


def inference(model,data_loader_test):
    """ Returns the inference accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    for batch_idx, (images, labels) in enumerate(data_loader_test):
        images, labels = images, labels

        # Inference
        outputs = model(images)

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    print(accuracy)

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
                               train=True,
                               download=True)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,

                                                    batch_size=64,
                                                    shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=64,
                                                   shuffle=True)
    model = Mnist_Model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                momentum=0.5)
    criterion = nn.NLLLoss()
    for iter in range(20):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(data_loader_train):
            model.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            inference()
