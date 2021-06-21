import numpy as np
import numpy as np
import os
import argparse


def make_entity2id_dir(path_dir):
    folders = [name for name in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, name))]

    for folder in folders:
        make_entity2id(path_dir + folder + "/")


def make_entity2id(path_dir):
    train_data_file = path_dir + "train.txt"
    test_data_file = path_dir + "test.txt"
    valid_data_file = path_dir + "valid.txt"

    print("processing :", path_dir)

    train = np.loadtxt(open(train_data_file, "rb"), delimiter="\t", skiprows=0, dtype="str", encoding="utf-8")
    test = np.loadtxt(open(test_data_file, "rb"), delimiter="\t", skiprows=0, dtype="str", encoding="utf-8")
    valid = np.loadtxt(open(valid_data_file, "rb"), delimiter="\t", skiprows=0, dtype="str", encoding="utf-8")
    train = np.concatenate((train, valid, test), axis=0)

    entity2id_ = dict()
    relation2id_ = dict()
    e_counter = 0
    r_counter = 0
    for row in range(0, train.shape[0]):

        if entity2id_.get(train[row][0], -1) == -1:
            entity2id_[train[row][0]] = e_counter
            e_counter = e_counter + 1
        if entity2id_.get(train[row][2], -1) == -1:
            entity2id_[train[row][2]] = e_counter
            e_counter = e_counter + 1
        if relation2id_.get(train[row][1], -1) == -1:
            relation2id_[train[row][1]] = r_counter
            r_counter = r_counter + 1

    entity2id_list = []
    for en in entity2id_.keys():
        entity2id_list.append([en, entity2id_[en]])

    rel2id_list = []
    for en in relation2id_.keys():
        rel2id_list.append([en, relation2id_[en]])

    np.savetxt(path_dir + "entity2id.txt", np.array(entity2id_list), fmt='%s', delimiter="\t",
               header=str(len(entity2id_list)))
    np.savetxt(path_dir + "relation2id.txt", np.array(rel2id_list), fmt='%s', delimiter="\t",
               header=str(len(rel2id_list)))


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Generating graph features for datasets',
        usage='python  extract_graph_features.py --path <path to dataset folder>'
    )
    parser.add_argument('--path', default=None, type=str, help='path to dataset folder')
    return parser.parse_args(args)


args = parse_args()
if args.path is not None:
    path = args.path
    make_entity2id_dir(path)
else:
    print("this file assumes that triples are in shape of h r t and they are tab separated")
    print("set --path for example ../data/fb15kp/")

# example  "--path ../data/wn18p/"
