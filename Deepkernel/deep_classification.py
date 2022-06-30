# -*- coding: utf-8 -*-

import os
import csv
import Module
import argparse
import numpy as np
import classification

def parse_args():
    '''
    Parses the SGN classification arguments.
    '''
    parser = argparse.ArgumentParser(description="Run DeepKernel.")

    datanum = '1'   # Dataset ID
    Type = 'MSGN'  # TSGN DTSGN TTSGN MSGN

    # Deep Kernel
    parser.add_argument('--oriemb', nargs='?', default=
    os.path.join('Emb_kernel', 'Data' + datanum + '_deepkernel_TN.csv'), help='TN Embedding path')

    parser.add_argument('--sgnemb', nargs='?', default=
    os.path.join('Emb_kernel', 'Data' + datanum + '_deepkernel_' + Type + '.csv'), help='SGN Embedding path')

    parser.add_argument('--label', nargs='?', default=
    os.path.join('Dataset', 'label' + datanum + '.Label'), help='Label path')
    parser.add_argument('--type', nargs='?',type=str,default=Type)
    parser.add_argument('--txtname', nargs='?', default='Data{}_deepkernel_{}.txt'.format(datanum, Type))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print('start reading features!')
    labels = Module.read_label(args.label)
    # np.set_printoptions(threshold=np.inf)
    total=[]

    if args.type == 'TN':
        with open(args.oriemb,'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            data1=[]
            for i in data:
                data1.append(map(float,i))
        data1=np.array(data1)

        print('classification begin!')

        for rate in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]: #
            out = classification.result_class(data, labels, split_rate=rate)
            total.append(out)


    else:
        with open(args.oriemb,'r') as f:
            reader = csv.reader(f)
            data = list(reader)

        with open(args.sgnemb,'r') as f:
            reader = csv.reader(f)
            dataa = list(reader)
        data1=[]
        data2=[]
        for i in range(0, 700):
            data1.append(data[i]+dataa[i])
        for i in data1:
            data2.append(map(float, i))

        data2=np.array(data2)
        print('classification begin!')

        for rate in [0.9]: #0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
            out = classification.result_class(data1, labels, split_rate=rate)
            total.append(out)

    print('The total final output:' )
    txt_name = args.txtname
    f = open(txt_name, "w")
    for i in total:
        f.write(i + "\n")
    f.close()
