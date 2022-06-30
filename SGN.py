# -*- coding: utf-8 -*-

import os
import time
import math
import shutil
import Module
import argparse
import networkx as nx
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="SGN")

    datanum = '1'   # Dataset ID
    Type = 'TTSGN'    # TSGN, DTSGN, TTSGN, MSGN

    parser.add_argument('--type', nargs='?', default=
    Type, help='Input graph path')

    parser.add_argument('--input', nargs='?', default=
    os.path.join('Dataset', "Ethereum" + datanum + "_TN"), help='Input graph path')

    parser.add_argument('--label', nargs='?', default=
    os.path.join('Dataset', 'label' + datanum + '.Label'), help='Label path')

    parser.add_argument('--sgnpath', nargs='?', default=
    os.path.join('Dataset', "Ethereum" + datanum + "_" + Type), help='SGN path')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if os.path.exists(args.sgnpath):
        shutil.rmtree(args.sgnpath)
    os.mkdir(args.sgnpath)

    files = os.listdir(args.input)
    files.sort(key=lambda x: int(x.split('.')[0]))

    t1 = time.time()
    print("\nmapping to sgn begin!\n")
    for path in tqdm(files):
        origin_path = os.path.join(args.input, path)
        sgn_path = os.path.join(args.sgnpath, path)

        # TSGN
        if args.type == 'TSGN':
            G = Module.read_graph(path=origin_path, weight=True, direct=False,type='TN')
            line = Module.to_line(graph=G, weight=True, conversion=False)

        # DTSGN
        elif args.type == 'DTSGN':
            G = Module.read_graph(path=origin_path, weight=True, direct=True,type='TN')
            line = Module.TSGN(G, conversion=False)

        # TTSGN
        elif args.type == 'TTSGN':
            G=Module.read_Tgraph(path=origin_path, multiple=False)
            line = Module.TTSGN(G, conversion=False)

        # MSGN
        elif args.type == 'MSGN':
            G=Module.read_Tgraph(path=origin_path,multiple=True)
            center=Module.read_center_address(path=origin_path)
            line=Module.MSGN(graph=G, center_address=center)

        # isolated node
        for node in line.nodes():
            if line.degree(node) == 0:
                if args.type == 'MSGN':
                    weights = nx.get_node_attributes(line, 'weight')
                    if (weights[node] == 0):
                        line.add_edge(node, node, weight=0)
                    else:
                        line.add_edge(node, node, weight=math.log10(weights[node]))
                else:
                    if (G[node[0]][node[1]]['weight'] == 0):
                        line.add_edge(node, node, weight=0)
                    else:
                        line.add_edge(node, node, weight=math.log10(G[node[0]][node[1]]['weight']))


        line = nx.convert_node_labels_to_integers(line, first_label=0, ordering='default')
        Module.write_graph_to_csv(sgn_path, line)

    t2 = time.time()
    print(f"Time consumption:{t2-t1}")
