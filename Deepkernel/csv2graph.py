# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx


### read edge list to networkx ###
def read_labelfile(Label):
	singlelist = []
	tag = []
	with open(Label) as f:
		for line in f.readlines():
			singlelist.append(line.strip()[-1])
			tag.append(line.split(" ")[1])
		labels = np.array(singlelist, dtype=np.uint16)

	return labels, tag


datanum = '1'
Type = 'MSGN'  # "TN", 'TSGN', 'DTSGN', 'TTSGN', 'MTSGN'
data_name = "Ethereum{}_{}".format(datanum, Type)

Label = r'./Dataset/label{}.Label'.format(data_name.split('_')[0][-1])  # for linux
# Label = r'.\Dataset\label{}.Label'.format(data_name.split('_')[0][-1])  # for windows
labels, tag = read_labelfile(Label)
filepaths = os.path.join('.', 'Dataset', data_name, '')
output_path = os.path.join('Graph_data', '{}_Ethereum{}.graph'.format(Type, data_name.split('_')[0][-1]))

if not os.path.exists('Graph_data'):
	os.mkdir('Graph_data')

data = {}
graph = {}

for i in tqdm(tag):
	G = nx.Graph()
	filepath = filepaths + f"{i}.csv"

	if Type == 'TN':
		DATA = pd.read_csv(filepath)  # for 'TN'
	else:
		DATA = pd.read_csv(filepath, names=["from", "to", "value"])  # for 'TSGN', 'DTSGN', 'TTSGN', and 'MTSGN'

	FROM = DATA['from'].tolist()
	TO = DATA['to'].tolist()
	VALUE = DATA['value'].tolist()

	# the rule of wl-test is only for undirected unweighted network
	for j in range(len(FROM)):
		G.add_edge(FROM[j], TO[j], weight=VALUE[j])

	G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

	nodes = {}
	for item in nx.degree(G):
		'''
		The node labels should be set as follow: 
		TN: degree of node
		TSGN: the weight of edge in TN
		DTSGN: the weight of edge in TN
		BUT set it as the degree of node in their network.
		'''
		G.add_node(item[0], label=(item[1],))  # set the attributes of nodes set label_number.append(first_label)
		G.add_node(item[0], neighbors=np.array(list(G.neighbors(item[0])), dtype=np.uint8))
		nodes[item[0]] = G.nodes[item[0]]

	i = int(i)
	graph[i] = nodes


data["graph"] = graph
data["labels"] = labels

if os.path.exists(output_path):
	os.remove(output_path)

output = open(output_path, 'wb')
pickle.dump(data, output,  pickle.HIGHEST_PROTOCOL)
output.close()
