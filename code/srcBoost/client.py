import torch
import copy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import Message
import numpy as np
import random
import time


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
    def __init__(self, args, dataset, data_idxs, speed, client_id, client_emb):
        self.args = args
        self.dataset = dataset
        self.trainloader, self.validloader, self.testloader, self.gradloader = self.train_val_test(
            dataset, list(data_idxs))
        self.dataset_count = {}
        # for i in range(10):
        #     self.dataset_count[i] = 0
        # for batch_idx, (images, labels) in enumerate(self.trainloader):
        #     for i in labels:
        #         self.dataset_count[int(i)] += 1
        self.device = 'cuda' if args.gpu else 'cpu'
        self.speed = speed
        self.id = client_id
        self.client_emb = client_emb
        self.trunc_loss = 0.0
    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]
        gradloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=int(51200000), shuffle=True)
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(512), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(512), shuffle=False)
        return trainloader, validloader, testloader, gradloader

    def update_weights(self, args, model, global_weight, now_time):
        message_list = []
        now_time = now_time
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
        end_flag = False
        use_time = 0.0
        limit_time = args.t1

        #client_grad_list = []
        print(f"train:{len(self.trainloader)}")
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                end_flag = False
                start_time = time.time()
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                # model_weights_grad = torch.ones(0)
                # for para in model.parameters():
                #     b = torch.flatten(para.grad)
                #     model_weights_grad = torch.concat((model_weights_grad, b))
                # client_grad_list.append(model_weights_grad)
                optimizer.step()
                batch_loss.append(loss.item())
                end_time = time.time()
                use_time += self.compute_use_time(end_time - start_time)
                if  use_time > limit_time:  #截断询问
                    limit_time += (args.t2 + args.t1)
                    if iter >= args.limit_ep: #条件1
                        upload_time = self.compute_use_time(mode='upload')
                        message_list.append(Message(client_id=self.id,
                                                    model_weight=copy.deepcopy(model.state_dict()),
                                                    start_time=now_time + use_time,
                                                    finish_upload_time=now_time + use_time + upload_time,
                                                    emb_distance=self.client_emb,
                                                    trunc_ep=iter))
                        end_flag = True
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if end_flag == False:
            model_weights_grad = torch.ones(0)
            for para in model.parameters():
                b = torch.flatten(para.grad)
                model_weights_grad = torch.concat((model_weights_grad, b))
            upload_time = self.compute_use_time(mode='upload')
            message_list.append(Message(client_id=self.id,
                                        model_weight=copy.deepcopy(model.state_dict()),
                                        start_time=now_time + use_time,
                                        finish_upload_time=now_time + use_time + upload_time,
                                        emb_distance=self.client_emb,
                                        trunc_ep=self.args.local_ep))
        return message_list

    def compute_use_time(self, use_time=0.0, mode='local_train'):
        if mode == 'local_train':
            if self.speed == 'fast':
                use_time *= (120 + random.expovariate(2))
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

        accuracy = correct / total
        return accuracy, loss

    def __lt__(self, other):
        return self.time < other.time


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    print("len:")
    print(len(test_dataset))
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128000,
                            shuffle=False)
    for batch_idx, (images, labels) in enumerate(testloader):
        print("test_inference")
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        batch_loss.backward()
        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    accuracy = correct / total
    return accuracy, loss
