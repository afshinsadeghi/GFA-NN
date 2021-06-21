import numpy as np
import os
import argparse


def covert_files_in_folder(path, output_path):
    print("processing :", path)
    entity2id = np.loadtxt(open(path + "entity2id.txt", "rb"), delimiter="\t", skiprows=0, dtype="str",
                           encoding="utf-8")
    entity2id_ = entity2id.copy()
    entity2id_[:, 0] = entity2id[:, 1]
    entity2id_[:, 1] = entity2id[:, 0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.savetxt(output_path + "entities.dict", entity2id_, fmt='%s', delimiter="\t")

    rel2id = np.loadtxt(open(path + "relation2id.txt", "rb"), delimiter="\t", skiprows=0, dtype="str", encoding="utf-8")
    if len(rel2id.shape) == 1:
        rel2id = np.expand_dims(rel2id, axis=0)
    rel2id_ = rel2id.copy()

    rel2id_[:, 0] = rel2id[:, 1]
    rel2id_[:, 1] = rel2id[:, 0]
    np.savetxt(output_path + "relations.dict", rel2id_, fmt='%s', delimiter="\t")


def make_dics(input_path, output_path):
    folders = [name for name in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, name))]

    for folder in folders:
        covert_files_in_folder(input_path + folder + "/", output_path + folder + "/")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Generating graph features for datasets',
        usage='python  extract_graph_features.py --path <path to dataset folder>'
    )
    parser.add_argument('--path', default="../data/", type=str, help='path to dataset folder')
    return parser.parse_args(args)


print("this file uses relation2id.txt and entity2id.txt and assumes they are tab separated")
print("set --path or defualt is ../data/wn18p/ and ../data/fb15kp/")
args = parse_args()
if args.path is not None:
    path = args.path
    make_dics(path, path)
else:
    input_path = "../data/wn18p/"
    output_path = "../data/wn18p/"
    make_dics(input_path, output_path)

    input_path = "../data/fb15kp/"
    output_path = "../data/fb15kp/"
    make_dics(input_path, output_path)

# example run --path ../data/wn18p/
