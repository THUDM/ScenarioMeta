import random, os
import torch
import torch.nn as nn
import numpy as np
import json

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def serialize(obj, path, in_json=False):
    if isinstance(obj, np.ndarray):
        np.save(path, obj)
    elif in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)


def unserialize(path):
    suffix = os.path.basename(path).split(".")[-1]
    if suffix == "npy":
        return np.load(path)
    elif suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return _pickle.load(file)


# pad a tensor at given dimension
def pad_tensor(tensor: torch.Tensor, length, value=0, dim=0) -> torch.Tensor:
    return torch.cat(
        (tensor, tensor.new_full((*tensor.size()[:dim], length - tensor.size(dim), *tensor.size()[dim + 1:]), value)),
        dim=dim)


# transform the list of list to tensor with possible padding
def list2tensor(data_list: list, padding_idx, dtype=torch.long, device=torch.device("cpu")):
    max_len = max(map(len, data_list))
    max_len = max(max_len, 1)
    data_tensor = torch.stack(
        tuple(pad_tensor(torch.tensor(data, dtype=dtype), max_len, padding_idx, 0) for data in data_list)).to(
        device)
    return data_tensor


def divide_dataset(dataset, valid_ratio=0.1, test_ratio=0.1):
    train_data, valid_data, test_data = [], [], []
    index = list(range(len(dataset)))
    random.shuffle(index)
    valid_size, test_size = round(len(dataset) * valid_ratio), round(len(dataset) * test_ratio)
    valid_index, test_index = set(index[:valid_size]), set(index[valid_size: valid_size + test_size])
    for i, data in enumerate(dataset):
        if i in valid_index:
            valid_data.append(data)
        elif i in test_index:
            test_data.append(data)
        else:
            train_data.append(data)
    return train_data, valid_data, test_data


class UserItemEmbeds(nn.Module):
    def __init__(self, user_embeds, item_embeds):
        nn.Module.__init__(self)
        self.user_embeds = user_embeds
        self.item_embeds = item_embeds

    def forward(self, nodes, neighbors=None, degrees=None, is_user=True, with_neighbor=True):
        if is_user:
            if with_neighbor and neighbors is not None and degrees is not None:
                return self.user_embeds(nodes), self.item_embeds(neighbors), degrees
            else:
                return (self.user_embeds(nodes),)
        else:
            if with_neighbor and neighbors is not None and degrees is not None:
                return self.item_embeds(nodes), self.user_embeds(neighbors), degrees
            else:
                return (self.item_embeds(nodes),)


class NeighborDict(nn.Module):
    def __init__(self, neighbor_dict=None, max_degree=512, padding_idx=0):
        nn.Module.__init__(self)
        self.neighbor_dict = neighbor_dict
        self.max_degree = max_degree
        self.flag = nn.Parameter(torch.empty(0), requires_grad=False)
        self.padding_idx = padding_idx

    def forward(self, nodes):
        if torch.is_tensor(nodes):
            if self.neighbor_dict is not None:
                neighbors = [random.sample(self.neighbor_dict[idx.item()], self.max_degree) if len(
                    self.neighbor_dict[idx.item()]) > self.max_degree else self.neighbor_dict[idx.item()] for idx in
                             nodes]
        else:
            if self.neighbor_dict is not None:
                neighbors = [random.sample(self.neighbor_dict[idx], self.max_degree) if len(
                    self.neighbor_dict[idx]) > self.max_degree else self.neighbor_dict[idx] for idx in nodes]
            nodes = torch.tensor(nodes, dtype=torch.long, device=self.flag.device)
        if self.neighbor_dict is not None:
            degrees = torch.tensor(list(map(len, neighbors)), dtype=torch.long, device=self.flag.device)
            neighbors = list2tensor(neighbors, self.padding_idx, device=self.flag.device)
            return nodes, neighbors, degrees
        else:
            return (nodes,)


def activation_method(name):
    """
    :param name: (str)
    :return: torch.nn.Module
    """
    name = name.lower()
    if name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    else:
        return nn.Sequential()


def filter_statedict(module):
    state_dict = module.state_dict(keep_vars=True)
    non_params = []
    for key, value in state_dict.items():
        if not value.requires_grad:
            non_params.append(key)
    state_dict = module.state_dict()
    for key in non_params:
        del state_dict[key]
    return state_dict
