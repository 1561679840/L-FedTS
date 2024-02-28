import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--task', type=str, default='FedAvg',
                    help="type of task")
    parser.add_argument('--epochs', type=int, default=75,
                        help="number of rounds of training")
    parser.add_argument('--lag_tolerance', type=int, default=4,
                        help="number of rounds of training")
    parser.add_argument('--num_clients', type=int, default=190,
                        help="number of clients: K")
    parser.add_argument('--time_limit', type=int, default=100,
                        help="time_limit")
    parser.add_argument('--fast', type=float, default=0.1,
                    help='portation of fast clients')
    parser.add_argument('--normal', type=float, default=0.4,
                        help='portation of normal clients')
    parser.add_argument('--slow', type=float, default=0.5,
                        help='portation of slow clients')
    parser.add_argument('--frac', type=float, default=0.35,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # other arguments
    parser.add_argument('--dataset', type=str, default='kdd', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
