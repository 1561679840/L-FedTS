import time
import copy
from tqdm import tqdm
from models import Mnist_Model, Boston_Model, KDD_Model, CIFAR_Model_ZFNet, Fmnist_BNNet
from utils import get_dataset, t1_selection, t2_selection, aggregation_weight, test_inference, get_acc_loss_grad
import torch
from collections import defaultdict
from options import args_parser
from client import Client
import numpy as np
import random
import lightgbm
def setup_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)


if __name__ == '__main__':
    for i in range(0,2):
        start_time = time.time()

        args = args_parser()
        setup_seed(args)
        if args.gpu:
            torch.cuda.set_device(int(args.gpu))
        device = 'cuda' if args.gpu else 'cpu'

        if args.dataset == 'mnist':
            global_model = Mnist_Model()
        elif args.dataset == 'fmnist':
            global_model = Fmnist_BNNet()
        elif args.dataset == 'cifar':
            global_model = CIFAR_Model_ZFNet()
        elif args.dataset == 'boston':
            global_model = Boston_Model()
        elif args.dataset == 'kdd':
            global_model = KDD_Model()
        else:
            raise NotImplementedError

        # load dataset and global model
        train_dataset, test_dataset, client_groups, client_emb = get_dataset(args)
        global_model.to(device)
        global_weights = copy.deepcopy(global_model.state_dict())

        # build clients
        fast_clients_num = args.num_clients * args.fast
        normal_clients_num = args.num_clients * args.normal
        slow_clients_num = args.num_clients - fast_clients_num - normal_clients_num
        client_list = []
        count_round = 0
        acc_list = []
        for i in range(args.num_clients):
            if i < fast_clients_num:
                speed = 'fast'
            elif i < fast_clients_num + normal_clients_num:
                speed = 'normal'
            else:
                speed = 'slow'
            client_list.append(Client(args, dataset=train_dataset, data_idxs=client_groups[i],
                                      speed=speed, client_id=i, client_emb=client_emb[i]))

        print(
            f"build {args.num_clients} success：{fast_clients_num} fast,{normal_clients_num} normal,{slow_clients_num} slow")
        print("start training:")

        client_message = defaultdict(list)
        for epoch in tqdm(range(args.epochs)):
            if epoch == 0:
                print("0:start distribution suceess!")
                for i in tqdm(range(args.num_clients)):
                    client_message[i] += client_list[i].update_weights(
                        args, global_model, copy.deepcopy(global_weights), now_time=0)
                    #debug
                    print([(message.start_time, message.finish_upload_time) for message in client_message[i]])
                    print([message.emb_distance for message in client_message[i]])
                print("O:model distribution suceess!")
            else:
                # message selection
                t1_message_list, rest_client_message = t1_selection(args, client_message, epoch)
                t1_pick_label = defaultdict(lambda :0)
                for message in t1_message_list:
                    label_dict = client_list[message.client_id].dataset_count
                    for key in label_dict:
                        t1_pick_label[key] += label_dict[key]
                #pre_global_weights
                pre_global_weights = aggregation_weight([message.model_weight
                                                         for message in t1_message_list])
                global_model.load_state_dict(pre_global_weights)
                #accuracy, loss, pre_global_weights_grad = test_inference(args, global_model, test_dataset)
                loss_dict = {}
                for client_id in range(args.num_clients):
                    message = rest_client_message[client_id]
                    if message:
                        gradloader = client_list[message[0].client_id].gradloader
                        _,loss = get_acc_loss_grad(args,global_model,gradloader)
                        loss_dict[client_id] = loss
                t2_message_list, rest_client_message = t2_selection(args,rest_client_message,loss_dict,epoch)
                t2_pick_label = defaultdict(lambda :0)
                for message in t2_message_list:
                    label_dict = client_list[message.client_id].dataset_count
                    for key in label_dict:
                        t2_pick_label[key] += label_dict[key]
                #t2_message_list, rest_client_message = t2_selection(args, t1_message_list, rest_client_message, epoch)
                client_message = rest_client_message
                global_weights = aggregation_weight([message.model_weight
                                                     for message in t1_message_list + t2_message_list])
                now_time = epoch * (args.t1 + args.t2)
                t1_client_id_set = set([message.client_id for message in t1_message_list])
                t2_client_id_set = set([message.client_id for message in t2_message_list])
                t1_client_ep_set = list([message.trunc_ep for message in t1_message_list])
                t2_client_ep_set = list([message.trunc_ep for message in t2_message_list])
                print(f"\nepoch:{epoch}\n,t1_time:{args.t1},t2_time:{args.t1}")
                print(f"t1_pick_message_len:{len(t1_message_list)}")
                for message in t1_message_list:
                  print(f"client_id:{message.client_id}, start:{message.start_time}, upload:{message.finish_upload_time},emb:{message.emb_distance}")
                print(f"t1_pick_client_len:{len(t1_client_id_set)}")
                print(f"t1_pick_client:{t1_client_id_set}")
                print(f"t2_pick_message_len:{len(t2_message_list)}")
                print(f"t2_pick_client_len:{len(t2_client_id_set)}")
                print(f"t2_pick_client:{t2_client_id_set}")
                for message in t2_message_list:
                  print(f"client_id:{message.client_id}, start:{message.start_time}, upload:{message.finish_upload_time},emb:{message.emb_distance}")

                print(f"time:{now_time}")
                print("evaluation...")
                global_model.load_state_dict(global_weights)
                accuracy, loss, _ = test_inference(args, global_model, test_dataset)
                print(f"accuracy:{accuracy}")
                print(f"loss:{loss}")
                count_round +=1
                print("====================")
                print(count_round)
                print("====================")
                if count_round % 1 == 0:
                    sum_acc = 0.0
                    for acc in acc_list:
                        sum_acc += acc
                    avg_acc = sum_acc / 1.0
                    count_round = 0
                    acc_list = []

                    with open("record.txt", 'a') as f:
                        f.write(f"\nepoch:{epoch},all_len:{len(t1_client_id_set)+len(t2_client_id_set)},accuracy:{accuracy},loss:{loss},t1_pick_message_len:{len(t1_message_list)},t1_pick_client_len:{len(t1_client_id_set)},t1_pick_client:{t1_client_id_set},t1_client_ep:{t1_client_ep_set},t2_pick_message_len:{len(t2_message_list)},t2_pick_client_len:{len(t2_client_id_set)},t2_pick_client:{t2_client_id_set},t2_client_ep:{t2_client_ep_set},t1_pick_label:{t1_pick_label},t2_pick_label:{t2_pick_label}")

                for client_id in tqdm((t1_client_id_set | t2_client_id_set)):
                    client_message[client_id] = list(filter(lambda x: x.start_time < epoch * (args.t1 + args.t2),
                                                            client_message[client_id]))

                    client_message[client_id] += client_list[client_id].update_weights(
                        args, global_model, copy.deepcopy(global_weights), now_time=epoch * (args.t1 + args.t2))
                    print([(message.start_time, message.finish_upload_time) for message in client_message[client_id]])
                    print([message.emb_distance for message in client_message[client_id]])

