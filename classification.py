# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from tqdm import tqdm
from random import randint

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# handcrafted features
def character(G):

    # The Number of Nodes
    d1 = G.number_of_nodes()
    if d1 == 0:
        return [0 for i in range(10)]
    # The Number of Edges
    d2 = G.number_of_edges()
    # Average Degree
    d3 = np.average([d for n, d in G.degree()])
    # Proportion of Leaf Nodes
    d4 = nx.degree_histogram(G)[0]
    # Average Clustering Coefficient
    d5 = nx.average_clustering(G)
    # c = nx.clustering(G).values()
    # d5 = sum(c)/len(c)

    # Maximum Eigenvalue of Adjacency Matrix
    d6 = np.max(nx.linalg.spectrum.adjacency_spectrum(G)).real
    # Network Density
    d7 = nx.density(G)
    # Betweenness Centrality
    d8 = np.average(list(nx.betweenness_centrality(G).values()))
    # Closeness Centrality
    d9 = np.average(list(nx.closeness_centrality(G).values()))

    # d10 = np.average(list(nx.eigenvector_centrality_numpy(G).values()))
    # Average Neighbor Degree
    d11 = np.average(list(nx.average_neighbor_degree(G).values()))

    chara = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d11]

    return chara


def RandomForest(X_train, X_test, Y_train, Y_test):
    param = {"n_estimators":[20, 50, 100, 200]}, #, "criterion":("gini", "entropy")}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=1),
                               n_jobs=-1, param_grid=param, cv=10, return_train_score=True)
    grid_search.fit(X_train, Y_train)
    Y_pred = grid_search.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    return acc


def perform_classification(X, Y, split_rate=0.8):      # X-feature Y-Label
    # starttime = time.time()
    seed = randint(0, 1000)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=split_rate, random_state=seed)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    Y_train = np.array([int(char) for char in Y_train])
    Y_test = np.array([int(char) for char in Y_test])


    test = RandomForest(X_train_std, X_test_std, Y_train, Y_test)
    return test


def result_class(X, Y, split_rate=0.8):    # X-feature Y-Label
    X = np.array(X)

    final = {}
    max = []
    for k in tqdm(range(30)):
        Acc = []
        for i in range(10):
            acc = perform_classification(X, Y,split_rate=split_rate)
            Acc.append(acc)
        ave = np.average(Acc)
        std = np.std(Acc)
        max.append(np.max(Acc))
        if ave not in final:
            final[ave] = std
        print("the test accuracy: {}, {}, {}".format(k, ave, std))
        print("the max accuracy every time: {}, {}".format(k, np.max(Acc)))

    print("the max accuracy & std: {}, {}".format(np.average(max), np.std(max)))
    acc = np.max(list(final.keys()))
    std = final[acc]
    print("the final accuracy: {}, {}".format(acc, std))
    return "the final accuracy: {}, {}".format(acc, std)