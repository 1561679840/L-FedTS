import copy
import torch
from torch import nn
from data import MnistData,BostonData,KddData,FMnistData,CIFARData
from torch.utils.data import DataLoader, Dataset
from models import hinge_loss
from collections import namedtuple

Message = namedtuple("Message", ['client_id', 'model_weight', 'start_time', 'finish_upload_time', 'emb_distance', 'trunc_ep'])


def get_dataset(args):
    """ Returns train and test datasets and a client group which is a dict where
    the keys are the client index and the values are the corresponding data for
    each of those clients.
    """

    if args.dataset == 'mnist':
        dataset = MnistData(args.num_clients)
    elif args.dataset == 'fmnist':
        dataset = FMnistData(args.num_clients)
    elif args.dataset == 'cifar':
        dataset = CIFARData(args.num_clients)
    elif args.dataset == 'boston':
        dataset = BostonData(args.num_clients)
    elif args.dataset == 'kdd':
        dataset = KddData(args.num_clients)
    else:
        return NotImplementedError

    return dataset.train_dataset, dataset.test_dataset, dataset.dict_clients, dataset.dict_clients_emb


def t1_selection(args, client_message, epoch):
    pick_message_list = []
    t1_time = (epoch - 1) * (args.t1 + args.t2) + args.t1
    for client_id in range(args.num_clients):
        pick_message_list += list(filter(lambda x: x.finish_upload_time <= t1_time, client_message[client_id]))
        client_message[client_id] = list(filter(lambda x: x.finish_upload_time > t1_time, client_message[client_id]))
    return pick_message_list, client_message


# def t2_selection(args, t1_message_list, client_message, epoch):
#     pick_message_list = []
#     t2_time = epoch * (args.t1 + args.t2)
#     if len(t1_message_list):
#         t1_emb_avg = sum([message.emb_distance for message in t1_message_list]) / len(t1_message_list)
#     else:
#         t1_emb_avg = 1000 #t1时刻没收到message，设置成比较大的emb距离
#     for client_id in range(args.num_clients):
#         pick_message_list += list(filter(
#             lambda x: x.finish_upload_time <= t2_time and x.emb_distance <= t1_emb_avg,
#             client_message[client_id]))
#         client_message[client_id] = list(filter(
#             lambda x: x.finish_upload_time > t2_time or x.emb_distance > t1_emb_avg,
#             client_message[client_id]))
#     return pick_message_list, client_message

# def t2_selection(args,pre_global_weights_grad,rest_client_message,epoch):
#     pick_message_list = []
#     t2_time = epoch * (args.t1 + args.t2)
#     Q = pre_global_weights_grad
#     for client_id in range(args.num_clients):
#         #计算剩余客户端与global grab之间的cos
#         pick_message_list += list(filter(
#             lambda K: K.finish_upload_time <= t2_time and torch.cosine_similarity(Q, K.client_grad, dim=0) <= 0,
#             rest_client_message[client_id]))
#         rest_client_message[client_id] = list(filter(
#             lambda K: K.finish_upload_time > t2_time and torch.cosine_similarity(Q, K.client_grad, dim=0) > 0,
#             rest_client_message[client_id]))
#     return pick_message_list, rest_client_message

def t2_selection(args,rest_client_message,loss_dict,epoch):
    pick_message_list = []
    t2_time = epoch * (args.t1 + args.t2)
    sort_loss_dict = sorted(loss_dict.items(), key= lambda d:d[1], reverse=True)
    print('======================')
    print(sort_loss_dict)
    id_list = []
    for key,value in sort_loss_dict:
        x = rest_client_message[key]
        y = x[0]
        if len(id_list)<30 and y.finish_upload_time <= t2_time:
            id_list.append(key)
    if epoch >=2:
        for client_id in range(args.num_clients):
            #计算剩余客户端与global grab之间的cos
            if client_id in id_list:
                pick_message_list += list(filter(
                    lambda K: K.finish_upload_time <= t2_time,
                    rest_client_message[client_id]))
            rest_client_message[client_id] = list(filter(
                lambda K: K.finish_upload_time > t2_time,
                rest_client_message[client_id]))
    else:
        for client_id in range(args.num_clients):
            #计算剩余客户端与global grab之间的cos
            if len(pick_message_list)<20:
                pick_message_list += list(filter(
                    lambda K: K.finish_upload_time <= t2_time,
                    rest_client_message[client_id]))
            rest_client_message[client_id] = list(filter(
                lambda K: K.finish_upload_time > t2_time,
                rest_client_message[client_id]))
    return pick_message_list, rest_client_message

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
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    model_weights_grad = torch.ones(0)
    for para in model.parameters():
        b = torch.flatten(para.grad)
        model_weights_grad = torch.concat((model_weights_grad, b))
    accuracy = correct / total
    return accuracy, loss, model_weights_grad

def get_acc_loss_grad(args, model, gradloader):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    for batch_idx, (images, labels) in enumerate(gradloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    accuracy = correct / total
    return accuracy, loss

