# -*- coding: utf-8 -*-

import os
import sys
import random
import scipy.io
import importlib
import numpy as np
import copy, time, math, pickle
from gensim.models import Word2Vec
from collections import defaultdict
importlib.reload(sys)
random.seed(314124)
np.random.seed(2312312)


def load_data(ds_name):
    f = open(os.path.join(DATA_DIR, "{}.graph".format(ds_name)), "rb")
    # print(DATA_DIR)
    # print(ds_name)
    data = pickle.load(f, encoding='bytes')
    # data = pickle.load(f)
    # print('data', data)
    graph_data = data['graph']
    labels = data["labels"]
    labels = np.array(labels, dtype = np.float)
    print("Saving the label data into: "+ OUTPUT_DIR + "/%s_labels.mat"%ds_name)
    scipy.io.savemat(OUTPUT_DIR + "/%s_labels.mat"%ds_name, mdict={'label': labels})
    return graph_data


def load_graph(nodes):
    max_size = []
    for i in nodes.keys():
        max_size.append(i)
    for nidx in nodes:
        for neighbor in nodes[nidx]["neighbors"]:
            max_size.append(neighbor)
    size = max(max_size)+1
    am = np.zeros((size, size))
    for nidx in nodes:
        for neighbor in nodes[nidx]["neighbors"]:
            am[nidx][neighbor] = 1
    return am


def get_maps(n):
    # canonical_map -> {canonical string id: {"graph", "idx", "n"}}
    file_counter = open("canonical_maps/canonical_map_n%s.p"%n, "rb")
    canonical_map = pickle.load(file_counter)
    file_counter.close()
    # weight map -> {parent id: {child1: weight1, ...}}
    file_counter = open("graphlet_counter_maps/graphlet_counter_nodebased_n%s.p"%n, "rb")
    weight_map = pickle.load(file_counter)
    file_counter.close()
    weight_map = {parent: {child: weight/float(sum(children.values())) for child, weight in children.items()} for parent, children in weight_map.items()}
    child_map = {}
    for parent, children in weight_map.items():
        for k,v in children.items():
            if k not in child_map:
                child_map[k] = {}
            child_map[k][parent] = v
    weight_map = child_map
    return canonical_map, weight_map


def adj_wrapper(g):
    am_ = g["al"]
    size = max(np.shape(am_))
    am = np.zeros((size, size))
    for idx, i in enumerate(am_):
        for j in i:
            am[idx][j-1] = 1
    return am


def build_wl_corpus(ds_name, max_h):
    graph_data = load_data(ds_name)
    #('graph_data:', len(graph_data), graph_data[1])
    labels = {}
    label_lookup = {}
    label_counter = 0
    vocabulary = set()
    num_graphs = len(graph_data)
    max_window = []
    wl_graph_map = {it: {gidx: defaultdict(lambda: 0) for gidx in list(graph_data)} for it in range(-1, max_h)}
    # print('wl_graph_map:', wl_graph_map)
    sim_map = {}

    # initial labeling
    # for gidx in range(num_graphs):
    for gidx in list(graph_data):
        gidx = int(gidx)
        labels[gidx] = np.zeros(len(graph_data[gidx]), dtype = np.int32)
        for node in range(len(graph_data[gidx])):
            label = graph_data[gidx][node]["label"]
            # if not label_lookup.has_key(label):
            if label not in label_lookup:
                label_lookup[label] = label_counter
                labels[gidx][node] = label_counter
                label_counter += 1
            else:
                labels[gidx][node] = label_lookup[label]
            wl_graph_map[-1][gidx][label_lookup[label]] = wl_graph_map[-1][gidx].get(label_lookup[label], 0) + 1
    compressed_labels = copy.deepcopy(labels)
    # WL iterations started
    for it in range(max_h):
        label_lookup = {}
        label_counter = 0
        for gidx in list(graph_data):
            gidx = int(gidx)
            for node in range(len(graph_data[gidx])):
                node_label = tuple([labels[gidx][node]])
                neighbors = graph_data[gidx][node]["neighbors"]
                if len(neighbors) > 0:
                    neighbors_label = tuple([labels[gidx][i] for i in neighbors])
                    node_label = tuple(tuple(node_label) + tuple(sorted(neighbors_label)))
                # if not label_lookup.has_key(node_label):
                if node_label not in label_lookup:
                    label_lookup[node_label] = str(label_counter)
                    compressed_labels[gidx][node] = str(label_counter)
                    label_counter += 1
                else:
                    compressed_labels[gidx][node] = label_lookup[node_label]
                wl_graph_map[it][gidx][label_lookup[node_label]] = wl_graph_map[it][gidx].get(label_lookup[node_label], 0) + 1
        # print('wl_graph_map:', wl_graph_map)
        print("Number of compressed labels at iteration %s: %s"%(it, len(label_lookup)))
        labels = copy.deepcopy(compressed_labels)

    # merge the following code into the loop above
    graphs = {}
    prob_map = {}
    corpus = []
    for it in range(-1, max_h):
        for gidx, label_map in wl_graph_map[it].items():
            if gidx not in graphs:
                graphs[gidx] = []
                prob_map[gidx] = {}
            for label_, count in label_map.items():
                label = str(it) + "+" + str(label_)
                for c in range(count):
                    graphs[gidx].append(label)
                vocabulary.add(label)
                prob_map[gidx][label] = count

    corpus = [graph for gidx, graph in graphs.items()]
    vocabulary = sorted(vocabulary)
    return corpus, vocabulary, prob_map, num_graphs, list(graph_data)


def l2_norm(vec):
    return np.sqrt(np.dot(vec, vec))


if __name__ == "__main__":
    # location to save the results
    OUTPUT_DIR = os.path.join("kernels")
    # location of the datasets
    # dataset = sys.argv[1]
    DATA_DIR = "Graph_data"
    DIR = 'deep_features'

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    # hyperparameters
    num_dimensions = 10    # int(sys.argv[1]) # any integer > 0
    kernel_type = 3        # int(sys.argv[2]) # 1 (deep, l2) or 2 (deep, M), 3 (MLE)
    feature_type = 3       # int(sys.argv[3]) # 1 (graphlet), 2 (SP), 3 (WL)
    # ds_name = sys.argv[4]  # dataset name
    window_size = 5        # int(sys.argv[5]) # any integer > 0

    # graph kernel parameters
    max_h = 2             # for WL

    # word2vec parameters
    ngram_type = 1      # int(sys.argv[6]) # 1 (skip-gram), 0 (cbow)
    sampling_type = 1   # int(sys.argv[7]) # 1 (hierarchical sampling), 0 (negative sampling)
    graphlet_size = 7   # int(sys.argv[8]) # any integer > 0
    sample_size = 100   # int(sys.argv[9]) # any integer > 0

    # print("Dataset: %s\n\nWord2vec Parameters:\nDimension: %s\nWindow size: %s\nNgram type: %s\nSampling type: %s\n\
    #         \n\nKernel-related Parameters:\nKernel type: %s\nFeature type: %s\nWL height: %s\nGraphlet size: %s\nSample size: %s\n"\
    #         %(ds_name, num_dimensions, window_size, ngram_type, sampling_type, kernel_type, feature_type, max_h, graphlet_size, sample_size))


    # STEP 1: Build corpus

    for file in os.listdir(DATA_DIR):
        # print(file)
        ds_name = file.split(".")[0]
        # ds_name = dataset + "_S_" + str(a) + "_" + str(b)
        # corpus, vocabulary, prob_map, num_graphs, graph_key = build_wl_corpus(ds_name, max_h)

        if ds_name == "MSGN_Ethereum2":
            corpus, vocabulary, prob_map, num_graphs, graph_key = build_wl_corpus(ds_name, max_h)
        else:
            continue

        # print('prob_map:', prob_map)

        vocabulary = list(sorted(vocabulary))
        # print("Corpus construction total time: %g vocabulary size: %s"%(end-start, len(vocabulary)))
        # STEP 2: learn hidden representations
        start = time.time()
        model = Word2Vec(corpus, size=num_dimensions, window=window_size,
                        min_count=0, sg=ngram_type, hs=sampling_type)
        end = time.time()
        print("M matrix total time: %g"%(end-start))

        # STEP 3: compute the kernel

        # 1.deep kernel

        H = []
        P = np.zeros((num_graphs, len(vocabulary)))
        # for i in range(num_graphs):
        #     for jdx, j in enumerate(vocabulary):
        #         P[i][jdx] = prob_map[i].get(j,0)

        for i in range(num_graphs):
            v = graph_key[i]
            for jdx, j in enumerate(vocabulary):
                P[i][jdx] = prob_map[v].get(j, 0)

        # M = np.zeros((len(vocabulary), len(vocabulary)))
        # for i in range(len(vocabulary)):
            # for j in range(len(vocabulary)):
                # M[i][j] = np.dot(model[vocabulary[i]], model[vocabulary[j]])
        # K = (P.dot(M)).dot(P.T)

        for i in vocabulary:
            print(model)
            H.append(model[i])
        H = np.array(H)
        K = P.dot(H)

        # 2.WL kernel

        # P = np.zeros((num_graphs, len(vocabulary)))
        # for i in range(num_graphs):
        #     v = graph_key[i]
        #     for jdx, j in enumerate(vocabulary):
        #         P[i][jdx] = prob_map[v].get(j, 0)
        #     # K = P.dot(P.T)
        # K = P

        # print(type(K), K.shape)
        np.savetxt(os.path.join(DIR, 'deep_kernel_{}_kernel_type_{}_feature_type_{}.csv'.format(ds_name, kernel_type, feature_type)), K, delimiter=',')
