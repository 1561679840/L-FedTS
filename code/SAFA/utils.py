import copy
import torch
import queue
import numpy as np
from torch import nn
from models import hinge_loss
from torch.utils.data import DataLoader
from data import MnistData,BostonData,KddData

def get_dataset(args):
    """ Returns train and test datasets and a client group which is a dict where
    the keys are the client index and the values are the corresponding data for
    each of those clients.
    """

    if args.dataset == 'mnist':
        dataset = MnistData(args.num_clients)
    elif args.dataset == 'boston':
        dataset = BostonData(args.num_clients)
    elif args.dataset == 'kdd':
        dataset = KddData(args.num_clients)
    else:
        return NotImplementedError

    return dataset.train_dataset, dataset.test_dataset, dataset.dict_clients, dataset.dict_clients_emb


def client_selection(now_time, limit, pick_num, client_list, last_pick_client_id):
    que1 = queue.PriorityQueue()
    que2 = queue.PriorityQueue()
    account = 0
    pick_client_id = []
    undraft_client_id = []
    for client in client_list:
        if client.now_time <= now_time + limit:
            que1.put(client)
            account += 1
    with open("recordl.txt", 'a') as f:
       f.write(
          f"\nacount:{account}")

    while que1.empty() == 0 and len(pick_client_id) < pick_num:
        client = que1.get()
        if client.id not in last_pick_client_id:
            pick_client_id.append(client.id)
        else:
            que2.put(client)

    while que2.empty() == 0 and len(pick_client_id) < pick_num:
        client = que2.get()
        pick_client_id.append(client.id)

    while que2.empty() == 0:
        client = que2.get()
        undraft_client_id.append(client.id)

    if len(pick_client_id) < pick_num:
        now_time = now_time + limit
    else:
        now_time = [client_list[id].now_time for id in pick_client_id + undraft_client_id]
        now_time = max(now_time)
    return now_time, pick_client_id, undraft_client_id


def aggregation_weight(caches):
    w_avg = copy.deepcopy(caches[0])
    for key in w_avg.keys():
        for i in range(1, len(caches)):
            w_avg[key] += caches[i][key]
        w_avg[key] = torch.div(w_avg[key], len(caches))
    return w_avg


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    if args.dataset=='kdd':
      criterion = hinge_loss
    else:
      criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss
