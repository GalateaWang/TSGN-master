# !/usr/bin/python3

import os
import time
import Module
import argparse
from tqdm import tqdm
import classification
import numpy as np


def parse_args():
    '''
    Parses the handcraft arguments.
    '''
    parser = argparse.ArgumentParser(description="Run Handcraft.")

    datanum = '1'   # Dataset ID
    Type = 'TSGN'   # TN TSGN DTSGN TTSGN

    # SGN
    parser.add_argument('--input', nargs='?', default=
    os.path.join('Dataset', "Ethereum" + datanum + '_' + Type), help='Input graph path')

    parser.add_argument('--emb', nargs='?', default=
    os.path.join('Emb', 'Data' + datanum + '_Handcraft_' + Type + '.pkl'), help='Feature path')

    parser.add_argument('--label', nargs='?', default=
    os.path.join('Dataset', 'label' + datanum + '.Label'), help='Label path')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_featrue = []
    fea_dict = {}
    files = os.listdir(args.input)
    files.sort(key=lambda x: int(x.split('.')[0]))

    t1 = time.time()
    for i, f in enumerate(tqdm(files)):
        path = os.path.join(args.input, f)
        G = Module.read_graph(path=path, weight=True, direct=True)
        chara = classification.character(G)
        all_featrue.append(chara)
        fea_dict[f.split('.')[0]] = chara
    t2 = time.time()
    print(f"Time Consumption:{t2-t1}")

    if os.path.exists(args.emb):
        os.remove(args.emb)

    Module.save_emb(fea_dict, path=args.emb)
    # labels = Module.read_label(args.label)
    all_featrue = np.array(all_featrue)

    # classification.result_class(all_featrue, labels)
