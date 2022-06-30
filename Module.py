# -*- coding: utf-8 -*-

import os
import csv
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def read_graph(path, weight=False, direct=False,type='TN'):
    """
    READ A GRAPH
    :param path: graph's csv path
    :param weight: if edge has the weight, (True or False)
    :param direct: if it is a directed graph, (True or False)
    :return:
    """
    cache = 'cache.csv'

    if(type=='TN'):
        data = pd.read_csv(path)
        data = list(data[['from', 'to', 'value']].values)

    if(type=='SGN'):
        data = pd.read_csv(path,header=None)
        data = list(data.values)

    if os.path.exists(cache):
        os.remove(cache)

    wtrie_list_to_csv(cache, data)

    G = nx.read_weighted_edgelist(cache, delimiter=',', \
                                  nodetype=str, create_using=nx.DiGraph())

    if not weight:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not direct:
        G = G.to_undirected()

    os.remove(cache)

    return G


def read_Tgraph(path, multiple=False):
    """
    read a graph with timestamp
    :param path: the path of graph
    :param multiple: if have multiple attributes
    :return:
    """
    data = pd.read_csv(path)
    data = data[['from', 'to', 'value', 'timestamp']]
    start_stamp = min(data['timestamp'].values)
    data['timestamp'] = data['timestamp'] - start_stamp
    if multiple:
        G = nx.MultiDiGraph()
        for index, item in data.iterrows():
            G.add_edge(item['from'], item['to'], weight=item['value'], timestamp=item['timestamp'])
    else:
        G = nx.DiGraph()
        for index, item in data.iterrows():
            G.add_edge(item['from'], item['to'], weight=item['value'], timestamp=item['timestamp'])

    return G


def read_center_address(path):
    """
    read center node's address
    :param path:
    :return: address
    """
    with open(path) as f:
        singlelist = [line.strip().split(' ')[0] for line in f.readlines()]
        address = np.array(singlelist)

    return address


def TSGN(graph, weight=True, directed=True, conversion=True):
    """
    extract TSGN from the origin
    :param graph: original graph
    :param weight: if DSGN have weight
    :param directed: if DSGN have direct
    :return: DSGN
    """
    G = nx.line_graph(graph)
    for edge in G.edges():
        # Update node weight
        w1 = graph[edge[0][0]][edge[0][1]]['weight']
        w2 = graph[edge[1][0]][edge[1][1]]['weight']
        # Update edge weights for SGN
        if w1 + w2 == 0:
            G[edge[0]][edge[1]]['weight'] = 0
        else:
            G[edge[0]][edge[1]]['weight'] = math.log10((w2 + w1) / 2)

    if conversion == True:
        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

    return G



def TTSGN(G, conversion=True):
    """
    extract TTSGN from the origin
    :param G: orginal graph
    :param conversion: if convert label
    return: TTSGN
    """
    graph = nx.line_graph(G)
    for edge in list(graph.edges()):
        if (G[edge[0][0]][edge[0][1]]['timestamp'] > G[edge[1][0]][edge[1][1]]['timestamp']):
            graph.remove_edge(edge[0], edge[1])

    for edge in graph.edges():
        # update node weight
        w1 = G[edge[0][0]][edge[0][1]]['weight']
        w2 = G[edge[1][0]][edge[1][1]]['weight']
        # update edge weights for SGN
        if w1 + w2 == 0:
            graph[edge[0]][edge[1]]['weight'] = 0
        else:
            graph[edge[0]][edge[1]]['weight'] = math.log10((w2 + w1) / 2)

    if conversion == True:
        graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default')

    return graph


def MSGN(graph, center_address, conversion=True, Threshold=0.3):
    """
    extract MSGN from original
    :param graph: original graph
    :param center_address: center node address
    :param conversion: if convert
    :param Threshold: The threshold for the percentage of gaps
    :return: MSGN
    """
    # target directed and weighted network
    G = nx.DiGraph()
    center_in = {}  # edges to a central node，like {address1:[(weight, timestamp),...]，address2:[(),...]}
    center_out = {}  # edges pointed out by the central node，like {address1:[(weight, timestamp),...]，address2:[(),...]}
    neibours_dict = {}  # Edges between first-order neighbors, like {address1:[(address2, weight, timestamp),...]，address3:[(),...]}
    threshold = gap_analysis(graph, center_address, threshold=Threshold)  # time interval threshold

    for edge in graph.edges(data=True):
        # build the target network's nodes
        G.add_node((edge[0], edge[1], edge[2]['timestamp']), weight=edge[2]['weight'])
        if threshold == None:
            continue
        # point out by the central node
        if edge[0] == center_address:
            if edge[1] not in center_out.keys():
                center_out[edge[1]] = [(edge[2]['weight'], edge[2]['timestamp'])]
            else:
                center_out[edge[1]].append((edge[2]['weight'], edge[2]['timestamp']))

        # to central node
        elif edge[1] == center_address:
            if edge[0] not in center_in.keys():
                center_in[edge[0]] = [(edge[2]['weight'], edge[2]['timestamp'])]
            else:
                center_in[edge[0]].append((edge[2]['weight'], edge[2]['timestamp']))

        # between neighbors
        else:
            if edge[0] not in neibours_dict.keys():
                neibours_dict[edge[0]] = [(edge[1], edge[2]['weight'], edge[2]['timestamp'])]
            else:
                neibours_dict[edge[0]].append((edge[1], edge[2]['weight'], edge[2]['timestamp']))

    if threshold == None:
        return G

    # First build a map around the central node
    for start_node in center_in.keys():
        for (weight1, timestamp1) in center_in[start_node]:
            for end_node in center_out.keys():
                for (weight2, timestamp2) in center_out[end_node]:
                    gap = timestamp2 - timestamp1
                    # if multiple edges
                    if start_node == end_node and -threshold <= gap < 0:
                        G.add_edge((center_address, end_node, timestamp2), (start_node, center_address, timestamp1),
                                   weight=weight_cal(weight1, weight2))

                    if gap > threshold or gap < 0:
                        continue
                    else:
                        G.add_edge((start_node, center_address, timestamp1), (center_address, end_node, timestamp2),
                                   weight=weight_cal(weight1, weight2))

    # Second, between the neighbor node and the central node
    for start_node in neibours_dict.keys():
        for (center_node, weight1, timestamp1) in neibours_dict[start_node]:
            if center_node in center_in.keys():
                for (weight2, timestamp2) in center_in[center_node]:
                    gap = timestamp2 - timestamp1
                    if gap > threshold or gap < 0:
                        continue
                    else:
                        G.add_edge((start_node, center_node, timestamp1), (center_node, center_address, timestamp2),
                                   weight=weight_cal(weight1, weight2))

            if start_node in center_out.keys():
                for (weight2, timestamp2) in center_out[start_node]:
                    gap = timestamp1 - timestamp2
                    if gap > threshold or gap < 0:
                        continue
                    else:
                        G.add_edge((center_address, start_node, timestamp2), (start_node, center_node, timestamp1),
                                   weight=weight_cal(weight1, weight2))

    # Last, between the neighbor nodes
            if center_node in neibours_dict.keys():
                for (end_node, weight2, timestamp2) in neibours_dict[center_node]:
                    gap = timestamp2 - timestamp1
                    # if multiple edges
                    if start_node == end_node and -threshold <= gap < 0:
                        G.add_edge((center_node, end_node, timestamp2), (start_node, center_node, timestamp1),
                                   weight=weight_cal(weight1, weight2))

                    if gap > threshold or gap < 0:
                        continue
                    else:
                        G.add_edge((start_node, center_node, timestamp1), (center_node, end_node, timestamp2),
                                   weight=weight_cal(weight1, weight2))

    if conversion == True:
        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

    return G


def gap_analysis(graph, center_address, threshold=0.3):
    """

    :param graph: multidigraph
    :param center_address： center_address
    :param threshold: percentage of total time intervals
    :return:
    """
    gap_list = []
    center_in = {}
    center_out = {}
    neibours_dict = {}

    for edge in graph.edges(data=True):
        # point by central node
        if edge[0] == center_address:
            timestamp = int(edge[2]["timestamp"])
            if edge[1] not in center_out.keys():
                center_out[edge[1]] = [timestamp]
            else:
                center_out[edge[1]].append(timestamp)

        # to central node
        elif edge[1] == center_address:
            timestamp = int(edge[2]["timestamp"])
            if edge[0] not in center_in.keys():
                center_in[edge[0]] = [timestamp]
            else:
                center_in[edge[0]].append(timestamp)

        else:
            if edge[0] not in neibours_dict.keys():
                neibours_dict[edge[0]] = [(edge[1], edge[2]["timestamp"])]
            else:
                neibours_dict[edge[0]].append((edge[1], edge[2]['timestamp']))

    # First count around the central node
    for startnode in center_in.keys():
        for timestamp1 in center_in[startnode]:
            for endnode in center_out.keys():
                for timestamp2 in center_out[endnode]:
                    if startnode == endnode:
                        gap_list.append(abs(timestamp1 - timestamp2))
                        continue
                    if timestamp2 - timestamp1 >= 0:
                        gap_list.append(timestamp2 - timestamp1)


    # Count the gap between neighbor nodes and central nodes, as well as between neighbor nodes。%neibours_dict：begin：[(end，timestamp)，...]%
    for startnode in neibours_dict.keys():
        for (endnode, timestamp) in neibours_dict[startnode]:
            # Gap between neighbor nodes and the center
            if startnode in center_out.keys():
                for to_start in center_out[startnode]:
                    if timestamp - to_start >= 0:
                        gap_list.append(timestamp - to_start)
            if endnode in center_in.keys():
                for out_end in center_in[endnode]:
                    if out_end - timestamp >= 0:
                        gap_list.append(out_end - timestamp)

            # gap between neighbors
            if endnode in neibours_dict.keys():
                for (neibour, nextstamp) in neibours_dict[endnode]:
                    if nextstamp - timestamp >= 0:
                        gap_list.append(nextstamp - timestamp)

    gap_list.sort()
    num_gap = len(gap_list)

    if num_gap == 0:
        return None
    else:
        return gap_list[math.ceil(threshold * num_gap) - 1]


def weight_cal(weight1, weight2):

    if weight1 + weight2 == 0:
        return 0
    else:
        return math.log10((weight1 + weight2) / 2)


def read_label(path):
    '''
    READ THE LABELS
    :param path: the label's path
    :return: np.array(labels)
    '''

    with open(path) as f:
        singlelist = [line.strip()[-1] for line in f.readlines()]
        labels = np.array(singlelist)

    return labels


def to_line(graph, weight=False, conversion=True):
    """
    turn the original graph to line graph.
    :param graph: original network (not directed)
    :param weight: whether the graph has weight
    :param conversion: whether convert node tick
    :return: linegraph
    """
    if not weight:
        graph_line = nx.line_graph(graph)
        if conversion:
            graph_line = nx.convert_node_labels_to_integers(graph_line, first_label=0, ordering='default')

    else:
        graph_line = nx.line_graph(graph)

        for edge in graph_line.edges():
            # print(edge)
            # print(edge[0][0], edge[0][1])

            w1 = graph[edge[0][0]][edge[0][1]]['weight']
            w2 = graph[edge[1][0]][edge[1][1]]['weight']

            if(w2 + w1)!=0:
                graph_line[edge[0]][edge[1]]['weight'] = math.log10((w2 + w1) / 2) #(w1 + w2) / 2
            else:
                graph_line[edge[0]][edge[1]]['weight'] = 0
        # print(graph_to_line)
        if conversion:
            graph_line = nx.convert_node_labels_to_integers(graph_line, first_label=0, ordering='default')

    return graph_line


def wtrie_list_to_csv(filepath, list):
    """
    write list to csv
    :param filepath:	the place to store the csv
    :param list: 	the list
    :return:
    """
    if os.path.exists(filepath):
        print(filepath)
    # os.rm(filepath)
    out = open(filepath, mode='w', encoding='utf8', newline='')
    csv_writer = csv.writer(out, dialect='excel')
    for x in list:
        csv_writer.writerow(x)
    out.close()


def write_graph_to_csv(filepath, graph):
    """
    write graph to csv
    :param filepath: the place to store csv
    :param graph: the graph which will be written
    :return: None
    """

    fea = [(edge[0], edge[1], graph[edge[0]][edge[1]]['weight']) for edge in graph.edges()]

    wtrie_list_to_csv(filepath, fea)


def save_emb(d, path):
    """
    save the graph embedding
    :param d: the feature dictionary
    :param path: the place where emb will be saved
    :return: None
    """
    if os.path.exists(path):
        os.remove(path)

    f = open(path, 'wb')
    pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def load_emb(path):
    """
    load the embedding
    :param path: the embedding path
    :return: dict
    """
    f = open(path, 'rb')
    res = pickle.load(f)
    f.close()
    return res


def dataset_info(path):
    """
    information of the dataset
    :param path: the path of dataset
    :return: None
    """
    edge_sum = 0
    node_sum = 0

    for i, CSV in enumerate(tqdm(os.listdir(path))):
        real_path = os.path.join(path, CSV)
        G = read_graph(real_path)
        node_sum += len(G.nodes)
        edge_sum += len(G.edges)

    count = i + 1
    print('*' * 50)
    print("DATASET NAME:{}".format(path.split('\\')[0]))
    print("The average number of nodes:")
    print(node_sum / count)
    print("The average number of edges:")
    print(edge_sum / count)
    print('*' * 50)


def plot_embedding(data):
    """
    max-min norm
    :param data: numpy data
    :return: after max-min norm
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    return data


def Tsne(feature, label):
    """
    show the Result after dimension reduce
    :param feature: the feature numpy
    :param label: the label numpy
    :return: None
    """

    feature = np.array(feature)
    label = np.array(label)
    tsne = TSNE(n_components=2, random_state=200, verbose=True).fit_transform(feature)
    tsne = plot_embedding(tsne)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=label, alpha=0.5)
    plt.show()
