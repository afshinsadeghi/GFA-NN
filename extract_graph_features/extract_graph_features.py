import networkx as nx
import numpy as np
import os
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Generating graph features for datasets',
        usage='python  extract_graph_features.py --path <path to dataset folder>'
    )
    parser.add_argument('--path', default="../data/", type=str, help='path to dataset folder')
    return parser.parse_args(args)


def store_graph_features(data_array, input_directory, dataset_name):
    print("extracting the 6 features for dataset in:", input_directory)
    MG = nx.MultiGraph()
    MG.add_weighted_edges_from(data_array, weight=1)
    MG_simple = nx.Graph(MG)

    dg = nx.degree(
        MG_simple)  # for  multi  graph gives the same address because it does not count relations between two nodes in calculating degree
    dg_per_graph = dg
    np.save(input_directory + dataset_name + "degree", dg_per_graph)

    length = dict(nx.all_pairs_dijkstra_path_length(MG_simple))
    np.save(input_directory + dataset_name + "lengthpaths", length)

    nodes = list(MG.nodes)
    number_of_random_nodes_paths = 3000
    random_nodes = np.random.choice(nodes, number_of_random_nodes_paths)
    random_paths_all = []
    for node1 in nodes:
        random_paths = []
        for node2 in random_nodes:
            path_ = length.get(node1, -1).get(node2, -1)
            random_paths.append(path_)
        random_paths_all.append(random_paths)

    random_paths_all = np.array(random_paths_all)

    np.save(input_directory + dataset_name + "selected_lengthpaths", random_paths_all)

    cen = nx.eigenvector_centrality_numpy(MG)  # gives dictionary of velaue per node
    np.save(input_directory + dataset_name + "centrality", cen)

    pg_rank = nx.pagerank(MG_simple, alpha=0.85)  # not implemented for muktigraphs
    np.save(input_directory + dataset_name + "pagerank", pg_rank)

    netw = nx.betweenness_centrality(MG_simple)
    np.save(input_directory + dataset_name + "betweenness", netw)

    katz = nx.katz_centrality_numpy(MG_simple)  # max_iter=1000000000  # not implemented for multigraph type
    np.save(input_directory + dataset_name + "katz", katz)


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    # indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def make_features_for_pattern_datasets(args):
    input_directory = args.path
    filenames = os.listdir(input_directory)  # get all files' and folders' names in the current directory

    print("extracting entity features related to datasets:", filenames)
    print("(please give folder of folders like ../data/ where ../data/WN18 is inside it)")
    result = []
    for filename in filenames:  # loop through all the files and folders
        if os.path.isdir(os.path.join(input_directory, filename)):  # check whether the current object is a folder
            result.append(filename)

    result.sort()

    for dataset_name in result:
        if not os.path.exists(input_directory + dataset_name + '/entities.dict'):
            print("entities.dict does not exist. first run makedict_for_pattern_rel_dbs.py to generate that. ")
            exit()

        with open(input_directory + dataset_name + '/entities.dict') as fin:
            entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        with open(input_directory + dataset_name + '/relations.dict') as fin:
            relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

        train_data_ = read_triple(input_directory + dataset_name + "/train.txt", entity2id, relation2id)
        train_data_ = np.array(train_data_)[:, [0, 2, 1]]  # it's column must be in shape [entity, entity,relation]

        out_dir_path = input_directory + dataset_name + "/train_node_features"
        if os.path.isdir(out_dir_path):
            print("Directory %s to create already exists." % out_dir_path)
        else:
            try:
                os.mkdir(out_dir_path)
            except OSError:
                print("Creation of the directory %s failed" % out_dir_path)
                exit()

        out_dir_path = out_dir_path + "/"
        store_graph_features(train_data_, out_dir_path, "")


print("example run: python extract_graph_features.py --path ../data/")
print("where the datasets like wn18rr etc are inside the folder ../data/")
make_features_for_pattern_datasets(parse_args())
