import sys
sys.path.append("..")
import networkx as nx
import numpy as np
import os
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
import torch
import Module
from tqdm import tqdm

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
np.random.seed(123)

########
def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


"""Adapted from https://github.com/weihua916/powerful-gnns/blob/master/util.py"""
class S2VGraph(object):
    def __init__(self, g, label, index, edge_mat=None, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.index = index
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = edge_mat
        self.max_neighbor = 0


def load_data(dataset, degree_as_tag=True, map_type=None):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        map_type: the type of map(TN,TSGN,DTSGN,TTSGN,MSGN)
        seed: random seed for random splitting of dataset
    '''
    data_path = os.path.join('../dataset/Dataset/', dataset)
    label_path = os.path.join('../dataset/Dataset/', dataset.split('/')[0] + '/label{}.Label'.format(dataset.split('_')[-2][-1]))
    labels = Module.read_label(label_path)
    num_class = len(set(labels))
    g_list = []
    for graph_id in tqdm(os.listdir(data_path)):
        id = graph_id.split('.')[0]
        edge_mat = []
        node_features = []
        label = labels[int(graph_id.split('.')[0]) - 1]

        if map_type == 'TSGN':
            G = Module.read_graph(os.path.join(data_path, graph_id), direct=False, weight=True, Type="SGN")
            G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
            for edge in G.edges():
                edge_mat.append([edge[0], edge[1]])
                edge_mat.append([edge[1], edge[0]])

        else:
            if map_type == 'TTSGN' or map_type == 'DTSGN':
                G = Module.read_graph(os.path.join(data_path, graph_id), direct=True, weight=True, Type="SGN")
            else:
                G = Module.read_graph(os.path.join(data_path, graph_id), direct=True, weight=True)
            G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

            for edge in G.edges():
                edge_mat.append([edge[0], edge[1]])

        edge_mat = np.array(edge_mat, dtype=np.int32)
        node_features = np.array(node_features, dtype=np.float32)
        edge_mat = edge_mat.T
        node_tags = list(dict(G.degree).values())

        g_list.append(S2VGraph(G, int(label), index=id, edge_mat=edge_mat, node_features=node_features, node_tags=node_tags))

    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = np.zeros((len(g.node_tags), len(tagset)), dtype=np.float32)
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    return g_list, num_class


def separate_data(graph_list, fold_idx, seed=0):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

"""Get indexes of train and test sets"""
def separate_data_idx(graph_list, fold_idx, seed=0):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    return train_idx, test_idx

"""Convert sparse matrix to tuple representation."""
def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
