import os
import time
import Module
import hashlib
import argparse
import networkx as nx
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def parse_args():
    '''
    Parses the Graph2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run Graph2vec.")

    datanum = '1'       # Dataset ID
    Type = 'TSGN'        # TN, TSGN, DTSGN, TTSGN, MSGN

    # SGN
    parser.add_argument('--input', nargs='?', default=
    os.path.join('Dataset', "Ethereum" + datanum + '_' + Type), help='Input graph path')

    parser.add_argument('--emb', nargs='?', default=
    os.path.join('Emb', 'Data' + datanum + '_Graph2vec_' + Type + '.pkl'), help='Feature path')

    parser.add_argument('--label', nargs='?', default=
    os.path.join('Dataset', 'label' + datanum + '.Label'), help='Label path')


    # graph2vec hyper parameters

    parser.add_argument("--dimensions", type=int, default=1024,
                        help="Number of dimensions. Default is 1024.")

    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers. Default is 4.")

    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of epochs. Default is 1000.")

    parser.add_argument("--min-count", type=int, default=5,
                        help="Minimal structural feature count. Default is 5.")

    parser.add_argument("--wl-iterations", type=int, default=2,
                        help="Number of Weisfeiler-Lehman iterations. Default is 2.")

    parser.add_argument("--learning-rate", type=float, default=0.025,
                        help="Initial learning rate. Default is 0.025.")

    parser.add_argument("--down-sampling", type=float, default=0.0001,
                        help="Down sampling rate of features. Default is 0.0001.")

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=True)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """

    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            # degs = [self.features[neb] for neb in nebs]
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()


def feature_extractor(graph, rounds, name):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """

    features = dict(nx.degree(graph))

    features = {k: v for k, v in features.items()}

    name = name.split('.')[0]
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc


def model(document_collections):
    print("\nOptimization started.\n")
    model = Doc2Vec(document_collections, vector_size=args.dimensions, window=0, min_count=args.min_count,
                    dm=0, sample=args.down_sampling, workers=args.workers,
                    epochs=args.epochs, alpha=args.learning_rate)
    return model


if __name__ == "__main__":

    args = parse_args()
    all_features = []
    fea_dict = {}
    doc = []
    files = os.listdir(args.input)
    files.sort(key=lambda x: int(x.split('.')[0]))

    t1 = time.time()
    print("\nfeature_extract start!\n")
    for path in tqdm(files):
        full_path = os.path.join(args.input, path)
        G = Module.read_graph(path=full_path, weight=args.weighted, direct=args.directed)
        fea = feature_extractor(G, args.wl_iterations, path)
        doc.append(fea)

    mymodel = model(doc)
    for path in tqdm(files):
        path = str(path.split('.')[0])
        l = mymodel.docvecs["g_" + path]
        all_features.append(l)
        fea_dict[path] = l

    t2 = time.time()
    print(f"Time Consumption:{t2-t1}")

    if os.path.exists(args.emb):
        os.remove(args.emb)

    Module.save_emb(d=fea_dict, path=args.emb)

    # labels = Module.read_label(args.label)

    # print("\nclassify start\n")
    # classification.result_class(all_features, labels)
