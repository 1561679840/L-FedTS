import time
import copy
from tqdm import tqdm
from models import Mnist_Model, Boston_Model, KDD_Model
from utils import get_dataset, client_selection, aggregation_weight, test_inference
import torch
from options import args_parser
from client import Client
import numpy as np
import random

def setup_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    setup_seed(args)
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    if args.dataset == 'mnist':
        global_model = Mnist_Model()
    elif args.dataset == 'boston':
        global_model = Boston_Model()
    elif args.dataset == 'kdd':
        global_model = KDD_Model()
    else:
        raise NotImplementedError

    # load dataset and global model
    train_dataset, test_dataset, client_groups,_ = get_dataset(args)
    global_model.to(device)
    global_weights = copy.deepcopy(global_model.state_dict())

    with open("record.txt", 'a') as f:
        f.write(
                f"\ntask:{args.task},dataset:{args.dataset},num_clients:{args.num_clients},frac:{args.frac},local_ep:{args.local_ep},slow:{args.slow}\n")



    # build clients
    fast_clients_num = args.num_clients * args.fast
    normal_clients_num = args.num_clients * args.normal
    slow_clients_num = args.num_clients - fast_clients_num - normal_clients_num
    client_list = []
    for i in range(args.num_clients):
        if i < fast_clients_num:
            speed = 'fast'
        elif i < fast_clients_num + normal_clients_num:
            speed = 'normal'
        else:
            speed = 'slow'
        client_list.append(Client(args, dataset=train_dataset, data_idxs=client_groups[i], speed=speed, client_id=i))

    print(
        f"build {args.num_clients} success：{fast_clients_num} fast,{normal_clients_num} normal,{slow_clients_num} slow")
    print("start training:")

    # epoch用作模型的版本，time用作时间
    bypass = []
    last_pick_client_id = []
    now_time = 0
    limit = args.time_limit
    pick_num = args.num_clients * args.frac
    count_round=0
    acc_list = []

    for epoch in tqdm(range(args.epochs)):
        if epoch == 0:
            print("0:start distribution suceess!")
            for i in tqdm(range(args.num_clients)):
                client_list[i].update_weights(global_model, copy.deepcopy(global_weights), version=epoch, now_time=now_time)
            print("O:model distribution suceess!")
        else:
            # client selection
            now_time, pick_client_id, undraft_client_id = client_selection(now_time, limit, pick_num,
                                                                       client_list, last_pick_client_id)

            print(f"\nepoch:{epoch}\n,pick:{pick_client_id},undraft:{undraft_client_id}")
            print(f"pick_clients_version:{[client_list[id].version for id in pick_client_id]}")
            print(f"pick_clients_time:{[client_list[id].now_time for id in pick_client_id]}")
            print(f"undraft_clients_version:{[client_list[id].version for id in undraft_client_id]}")
            print(f"undraft_clients_time:{[client_list[id].now_time for id in undraft_client_id]}")
            print(f"time:{now_time}")


            cache = [client_list[id].model_weight for id in pick_client_id]
            global_weights = aggregation_weight(cache + bypass)
            last_pick_client_id = pick_client_id
            bypass = [client_list[id].model_weight for id in undraft_client_id]
            for client in client_list:
                if client.id in pick_client_id + undraft_client_id:
                    client.update_weights(global_model, copy.deepcopy(global_weights), version=epoch, now_time=now_time)
                # elif client.version < epoch - args.lag_tolerance:
                #     client.update_weights(global_model, copy.deepcopy(global_weights), version=epoch, now_time=now_time)
            print("evaluation...")
            global_model.load_state_dict(global_weights)
            accuracy, loss = test_inference(args,global_model,test_dataset)
            print(f"accuracy:{accuracy}")
            print(f"loss:{loss}")
            acc_list.append(accuracy)
            count_round += 1
            print("====================")
            print(count_round)
            print("====================")
            if count_round % 5 == 0:
                sum_acc = 0.0
                for acc in acc_list:
                    sum_acc += acc
                avg_acc = sum_acc / 5
                count_round = 0
                acc_list = []

                with open("record.txt", 'a') as f:
                    f.write(
                        f"epoch:{epoch},accuracy:{avg_acc},loss:{loss},time:{now_time}\n")
