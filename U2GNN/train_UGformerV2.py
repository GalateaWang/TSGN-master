# -*- coding: utf-8 -*-

import os
import torch
torch.manual_seed(123)

import numpy as np
np.random.seed(123)
import time

from UGformerV2 import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# Parameters
# ==================================================

parser = ArgumentParser("UGformer", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="Dataset/Ethereum1_TN", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='TN', help="")
parser.add_argument("--dropout", default=0.5, type=float, help="")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="")
parser.add_argument("--nhead", default=1, type=int, help="")
parser.add_argument("--num_timesteps", default=1, type=int, help="Number of self-attention layers within each UGformer layer")
parser.add_argument("--ff_hidden_size", default=1024, type=int, help="The hidden size for the feedforward layer")
parser.add_argument('--fold_idx', type=int, default=1, help='The fold index. 0-9.')
parser.add_argument('--output', nargs='?', default=
f"../Emb/Data{1}_U2GNN_TN.pkl", help='The output pkl path')

args = parser.parse_args()

print(args)

# Load data
print("Loading data...")

use_degree_as_tag = False

graphs, num_classes = load_data(args.dataset, use_degree_as_tag, map_type=args.model_name)
train_graphs, test_graphs = separate_data(graphs, args.fold_idx)
feature_dim_size = graphs[0].node_features.shape[1]
print(feature_dim_size)
print('*' * 50)
print(args.dataset)

def get_Adj_matrix(graph):

    Adj_block_idx = torch.LongTensor(graph.edge_mat)
    Adj_block_elem = torch.ones(Adj_block_idx.shape[1])
    num_node = len(graph.g)

    if args.model_name == 'TN' or args.model_name == 'TSGN':
        self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
        elem = torch.ones(num_node)
        Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
        Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([num_node, num_node]))
    else:
        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([num_node, num_node]))

    return Adj_block.to(device) # can implement and tune for the re-normalized adjacency matrix D^-1/2AD^-1/2 or D^-1A like in GCN/SGC ???


def get_data(graph):
    node_features = graph.node_features
    node_features = torch.from_numpy(node_features).to(device)
    Adj_block = get_Adj_matrix(graph)
    graph_label = np.array([graph.label])
    graph_id = graph.index
    return Adj_block, node_features, torch.from_numpy(graph_label).to(device), graph_id

print("Loading data... finished!")


model = FullyConnectedGT_UGformerV2(feature_dim_size=feature_dim_size,
                        ff_hidden_size=args.ff_hidden_size,
                        num_classes=num_classes, dropout=args.dropout,
                        num_self_att_layers=args.num_timesteps,
                        num_GNN_layers=args.num_hidden_layers,
                        nhead=1).to(device) # nhead is set to 1 as the size of input feature vectors is odd

def cross_entropy(pred, soft_targets): # use nn.CrossEntropyLoss if not using soft labels in Line 159
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    idxs = np.arange(len(train_graphs))
    np.random.shuffle(idxs)
    for idx in idxs:
        Adj_block, node_features, graph_label, graph_id = get_data(train_graphs[idx]) # one graph per step. should modify to use "padding" (for node_features and Adj_block) within a batch size???
        # print(node_features.shape, Adj_block.shape)
        graph_label = label_smoothing(graph_label, num_classes)
        optimizer.zero_grad()
        prediction_score, graph_emb = model.forward(Adj_block, node_features)
        # loss = criterion(torch.unsqueeze(prediction_score, 0), graph_label)
        loss = cross_entropy(torch.unsqueeze(prediction_score, 0), graph_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # prevent the exploding gradient problem
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def evaluate():
    model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        # evaluating
        prediction_output = []
        for i in range(0, len(test_graphs)):
            Adj_block, node_features, graph_label, _ = get_data(test_graphs[i])
            prediction_score, _ = model.forward(Adj_block, node_features)
            prediction_score = prediction_score.detach()
            prediction_output.append(torch.unsqueeze(prediction_score, 0))
    prediction_output = torch.cat(prediction_output, 0)
    predictions = prediction_output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = predictions.eq(labels.view_as(predictions)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    return acc_test


"""main process"""
import os
out_dir = os.path.abspath(os.path.join(args.run_folder, "FullyConnectedGT_UGformerV2", args.dataset.split('/')[1]))
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
write_acc = open(checkpoint_prefix + '_acc.txt', 'w')

cost_loss = []
best_test_acc = 0
best_train_loss = float('inf')
emb = {}

for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    train_loss = train()
    cost_loss.append(train_loss)
    acc_test = evaluate()
    print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | test acc {:5.2f} | '.format(
                epoch, (time.time() - epoch_start_time), train_loss, acc_test*100))
    write_acc.write('epoch ' + str(epoch) + ' fold ' + str(args.fold_idx) + ' acc ' + str(acc_test*100) + '%\n')

    if train_loss <= best_train_loss and acc_test >= best_test_acc:
        best_test_acc = acc_test
        best_train_loss = train_loss

        for i in range(0, len(graphs)):
            Adj_block, node_features, graph_label, graph_id = get_data(graphs[i])
            _, graph_emb = model.forward(Adj_block, node_features)
            emb[graph_id] = graph_emb.detach().cpu().numpy()

Module.save_emb(d=emb, path=args.output)
write_acc.close()
