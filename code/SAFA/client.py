import torch
import copy
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import random

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class Client(object):
    def __init__(self, args, dataset, data_idxs, speed,client_id):
        self.args = args
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(data_idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.speed = speed
        self.id = client_id

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(512), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(512), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_weight,version,now_time):

        self.version = version
        self.now_time = now_time
        self.criterion = model.loss_fn

        model.load_state_dict(global_weight)
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                start_time = time.time()
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                end_time = time.time()
                self.now_time += self.compute_use_time(end_time - start_time)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        self.now_time += self.compute_use_time(mode='upload')
        self.model_weight = copy.deepcopy(model.state_dict())

    def compute_use_time(self, use_time=0.0, mode='local_train'):
        if mode == 'local_train':
            if self.speed == 'fast':
                use_time *= (120 + random.expovariate(2.0))
            elif self.speed == 'normal':
                use_time *= (200 + random.expovariate(1.5))
            elif self.speed == 'slow':
                use_time *= (280 + random.expovariate(0.8))
        else:
            use_time = 0.1 * random.lognormvariate(1.35, 0.35)
        return use_time

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

    def __lt__(self, other):
        return self.now_time < other.now_time



