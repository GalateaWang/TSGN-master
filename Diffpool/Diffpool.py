# -*- coding: utf-8 -*-

import os
import sys

sys.path.append("..")
import Module
import argparse
import numpy as np
import networkx as nx
import os.path as osp
from math import ceil
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.data import Dataset, Data, DenseDataLoader

max_nodes = 3000

def parse_args():
    parser = argparse.ArgumentParser(description="Run Diffpool")

    parser.add_argument('--input', nargs='?', default=
    "Ethereum2_TN", help='The input graph path')

    parser.add_argument('--output', nargs='?', default=
    f"Data{2}_Diffpool_TN.pkl", help='The output pkl path')

    parser.add_argument("--type", nargs='?', default='TN',
                        help="The type of map.")

    parser.add_argument("--num_class", type=int, default=2,
                        help="Number of classes. Default is 2.")

    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs. Default is 100.")

    return parser.parse_args()


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform

    @property
    def raw_file_names(self):
        return [args.input]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(i) for i in range(700)]

    @property
    def num_classes(self):
        return args.num_class

    def process(self):
        i = 0

        for raw_path in (self.raw_paths):

            path_list = os.listdir(raw_path)
            path_list.sort(key=lambda x: int(x.split('.')[0]))
            for csv_path in path_list:
                path = os.path.join(raw_path, csv_path)
                label_index = int(csv_path.split('.')[0])

                if args.type == 'TN' or args.type == 'MSGN':
                    G = Module.read_graph(path, weight=True, direct=True, type='TN')

                elif args.type == 'TSGN':
                    G = Module.read_graph(path, weight=True, direct=False,type='SGN')

                elif args.type == 'TTSGN' or args.type == 'DTSGN':
                    G = Module.read_graph(path, weight=True, direct=True,type='SGN')


                G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
                edge_index = []
                x = []
                y = []

                # extract the node feature x
                if args.type == 'TSGN':
                    for node in G.nodes():
                        x.append([G.nodes[node]['weight']])
                else:
                    for node in G.nodes():
                        x.append([G.in_degree(node, 'weight'), G.out_degree(node, 'weight')])

                if label_index % 2 == 0:
                    y.append(0)
                else:
                    y.append(1)

                # produce the edge list
                for edge in G.edges():
                    edge_index.append([edge[0], edge[1]])

                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

                data = Data(edge_index=edge_index, x=x, y=y, i=label_index)

                if self.pre_filter is not None:
                    pass

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


class ToDense(object):
    r"""Converts a sparse adjacency matrix to a dense adjacency matrix with
    shape :obj:`[num_nodes, num_nodes, *]`.

    Args:
        num_nodes (int): The number of nodes. If set to :obj:`None`, the number
            of nodes will get automatically inferred. (default: :obj:`None`)
    """

    def __init__(self, num_nodes=None):
        self.num_nodes = num_nodes

    def __call__(self, data):
        assert data.edge_index is not None

        orig_num_nodes = data.num_nodes
        if self.num_nodes is None:
            num_nodes = orig_num_nodes
        else:
            assert orig_num_nodes <= self.num_nodes
            num_nodes = self.num_nodes

        if data.edge_attr is None:
            edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float)
        else:
            edge_attr = data.edge_attr

        size = torch.Size([num_nodes, num_nodes] + list(edge_attr.size())[1:])
        adj = torch.sparse_coo_tensor(data.edge_index, edge_attr, size)
        data.adj = adj.to_dense()
        data.edge_index = None
        data.edge_attr = None

        data.mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.mask[:orig_num_nodes] = 1

        if data.x is not None:
            size = [num_nodes - data.x.size(0)] + list(data.x.size())[1:]
            data.x = torch.cat([data.x, data.x.new_zeros(size)], dim=0)

        if data.pos is not None:
            size = [num_nodes - data.pos.size(0)] + list(data.pos.size())[1:]
            data.pos = torch.cat([data.pos, data.pos.new_zeros(size)], dim=0)

        if data.y is not None and (data.y.size(0) == orig_num_nodes):
            data.y = data.y

        return data

    def __repr__(self):
        if self.num_nodes is None:
            return '{}()'.format(self.__class__.__name__)
        else:
            return '{}(num_nodes={})'.format(self.__class__.__name__,
                                             self.num_nodes)


args = parse_args()
dataset = MyOwnDataset('Ethereum', transform=ToDense(max_nodes))
dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10

test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=2)
val_loader = DenseDataLoader(val_dataset, batch_size=2)
train_loader = DenseDataLoader(train_dataset, batch_size=2)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)
        self.gnn1_embed = GNN(dataset.num_features, 64, 64, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, dataset.num_classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        embedding = x
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), embedding, l1 + l2, e1 + e2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
emb_res = {}
emb_now = {}


def train(epoch):
    model.train()
    loss_all = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, emb, _, _ = model(data.x, data.adj)
        loss = F.nll_loss(output, data.y.view(-1).long())
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()

    for data in tqdm(train_loader):
        data = data.to(device)
        output, emb, _, _ = model(data.x, data.adj)
        for num, e in enumerate(emb):
            emb_now[data.i[num].cpu().detach().item()] = np.squeeze(e.cpu().detach().numpy())

    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.adj)[0].max(dim=1)[1]
        emb = model(data.x, data.adj)[1]
        for num, e in enumerate(emb):
            emb_now[data.i[num].cpu().detach().item()] = np.squeeze(e.cpu().detach().numpy())
        y = data.y.view(-1)
        # TP    predict 和 label 同时为1
        TP += ((pred == 1) & (y == 1)).cpu().sum().item()
        # TN    predict 和 label 同时为0
        TN += ((pred == 0) & (y == 0)).cpu().sum().item()
        # FN    predict 0 label 1
        FN += ((pred == 0) & (y == 1)).cpu().sum().item()
        # FP    predict 1 label 0
        FP += ((pred == 1) & (y == 0)).cpu().sum().item()

    ze = 10e-7
    p = TP / (TP + FP + ze)
    r = TP / (TP + FN + ze)
    F1 = 2 * r * p / (r + p + ze)

    return F1


best_val_acc = test_acc = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        emb_res = emb_now
        test_acc = test(test_loader)
        best_val_acc = val_acc
        Module.save_emb(d=emb_res, path=args.output)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Val F1-score: {val_acc:.4f}, Test F1-score: {test_acc:.4f}')
