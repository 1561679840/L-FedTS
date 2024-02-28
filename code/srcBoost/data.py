import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import torch
from torch.utils.data import TensorDataset



def prob_dist(probs, probs_uniform=np.array([0.1] * 10), norm=True):
    # Given a list of probabilities in probs:
    # Norm==False: compute the sum of square of probability difference
    # Norm==True: compute the 1-norm of probability difference
    # Difference compared to a uniform distribution
    if norm:
        return np.linalg.norm(probs - probs_uniform, ord=1)
    else:
        return np.square(probs - probs_uniform).sum()


def noniid_unequal(dataset, num_clients, num_shards=1200, labels=None):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_clients:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = int(num_shards), int(len(dataset) / num_shards)
    #assert num_shards * num_imgs == len(dataset)
    idx_shard = [i for i in range(num_shards)]
    dict_clients = {i: np.array([]) for i in range(num_clients)}
    dict_clients_emb = {i: 0 for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    if labels is None:
        labels = dataset.train_labels.numpy()
    n_classes = labels.max() + 1
    label_dict = {i: np.array([]) for i in range(num_clients)}
    label_prob_dict = {i: np.zeros((n_classes,)) for i in range(num_clients)}

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]
    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_clients)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_clients):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_clients[i] = np.concatenate(
                    (dict_clients[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
                label_dict[i] = np.concatenate(
                    (label_dict[i], labels[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_clients):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_clients[i] = np.concatenate(
                    (dict_clients[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
                label_dict[i] = np.concatenate(
                    (label_dict[i], labels[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_clients):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_clients[i] = np.concatenate(
                    (dict_clients[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
                label_dict[i] = np.concatenate(
                    (label_dict[i], labels[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_clients, key=lambda x: len(dict_clients.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_clients[k] = np.concatenate(
                    (dict_clients[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
                label_dict[k] = np.concatenate(
                    (label_dict[k], labels[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    for i in range(num_clients):
        for label in label_dict[i]:
            label_prob_dict[i][int(label)] += 1
        label_prob_dict[i] = label_prob_dict[i] / label_prob_dict[i].sum()
        dict_clients_emb[i] = prob_dist(label_prob_dict[i], probs_uniform=np.array([1.0 / n_classes] * n_classes))
    return dict_clients, dict_clients_emb


class KddData(object):

    def __init__(self, num_clients):
        from sklearn import datasets
        kddcup99 = datasets.fetch_kddcup99()
        self._encoder = {
            'protocal': LabelEncoder(),
            'service': LabelEncoder(),
            'flag': LabelEncoder(),
            'label': LabelEncoder()
        }
        data_X, data_y = self.__encode_data(kddcup99.data, kddcup99.target)
        self.train_dataset, self.test_dataset = self.__split_data_to_tensor(data_X, data_y)
        self.dict_clients, self.dict_clients_emb = noniid_unequal(self.train_dataset,
                                                                  num_clients,
                                                                  num_shards=1453,
                                                                  labels=self.y_train)

    def __encode_data(self, data_X, data_y):
        self._encoder['protocal'].fit(list(set(data_X[:, 1])))
        self._encoder['service'].fit(list(set(data_X[:, 2])))
        self._encoder['flag'].fit((list(set(data_X[:, 3]))))
        self._encoder['label'].fit(list(set(data_y)))
        data_X[:, 1] = self._encoder['protocal'].transform(data_X[:, 1])
        data_X[:, 2] = self._encoder['service'].transform(data_X[:, 2])
        data_X[:, 3] = self._encoder['flag'].transform(data_X[:, 3])
        data_y = self._encoder['label'].transform(data_y)
        return data_X, data_y

    def __split_data_to_tensor(self, data_X, data_y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_X, data_y, test_size=0.2)
        train_dataset = TensorDataset(
            torch.from_numpy(self.X_train.astype(np.float32)),
            torch.from_numpy(self.y_train.astype(np.int))
        )
        test_dataset = TensorDataset(
            torch.from_numpy(self.X_test.astype(np.float32)),
            torch.from_numpy(self.y_test.astype(np.int))
        )
        return train_dataset, test_dataset


class MnistData(object):

    def __init__(self,num_clients):
        from torchvision import datasets, transforms
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        self.train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                            transform=apply_transform)

        self.test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                           transform=apply_transform)
        self.dict_clients, self.dict_clients_emb = noniid_unequal(self.train_dataset,
                                                                  num_clients,
                                                                  num_shards=1200)

class BostonData(object):
    def __init__(self,num_clients):
        from sklearn.datasets import load_boston
        boston_data = load_boston()
        self._encoder = {
            'label': LabelEncoder()
        }
        data_X, data_y = self.__encode_data(boston_data.data, boston_data.target)
        self.train_dataset, self.test_dataset = self.__split_data_to_tensor(data_X, data_y)
        self.dict_clients, self.dict_clients_emb = noniid_unequal(self.train_dataset,
                                                                  num_clients,
                                                                  num_shards=10,
                                                                  labels=self.y_train)


    def __encode_data(self,data_X, data_y):
        ss = MinMaxScaler()
        data_X = ss.fit_transform(data_X)
        print("Boston_label_cut")
        label_df = pd.DataFrame({'label':data_y})
        label_df['label_group'] = pd.cut(label_df['label'], 5)
        # 查看每个分组里变量的个数
        print(label_df['label_group'].value_counts())
        data_y = self._encoder['label'].fit_transform(label_df['label_group'].values)
        return data_X,data_y

    def __split_data_to_tensor(self, data_X, data_y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_X, data_y, test_size=0.15)
        train_dataset = TensorDataset(
            torch.from_numpy(self.X_train.astype(np.float32)),
            torch.from_numpy(self.y_train.astype(np.int))
        )
        test_dataset = TensorDataset(
            torch.from_numpy(self.X_test.astype(np.float32)),
            torch.from_numpy(self.y_test.astype(np.int))
        )
        return train_dataset, test_dataset
    
class FMnistData(object):

    def __init__(self,num_clients):
        from torchvision import datasets, transforms
        data_dir = '../data/fmnist/fmnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.], [0.5])])
        self.train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                            transform=apply_transform)

        self.test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                           transform=apply_transform)
        self.dict_clients, self.dict_clients_emb = noniid_unequal(self.train_dataset,
                                                                  num_clients,
                                                                  num_shards=1200)

class CIFARData(object):

    def __init__(self,num_clients):
        from torchvision import datasets, transforms
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.465,0.406],[0.229,0.224,0.225])])
        self.train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                            transform=apply_transform)
        self.test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                           transform=apply_transform)

        self.dict_clients, self.dict_clients_emb = noniid_unequal(self.train_dataset,
                                                                  num_clients,
                                                                  num_shards=1250,
                                                                  labels=np.array(self.train_dataset.targets))




