'''
This file is modified from 
https://github.com/NIRVANALAN/gcn_analysis/blob/master/notebook/Plantenoid%20Citation%20Data%20Format%20Transformation.ipynb

'''
import networkx as nx
import pickle as pkl
import pickle
import sys
from scipy.sparse.linalg.eigen.arpack import eigsh
import scipy.sparse as sp
import time
import argparse
import numpy as np
import pdb
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_features_and_labels(path, dataset, save_path):

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # features = normalize(features) # no normalization in plantoid


    labels = encode_onehot(idx_features_labels[:, -1])
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                    dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    pickle.dump(features[idx_train], open(f"{save_path}/ind.cora.x", "wb"))
    d = sp.vstack((features[:idx_test[0]], features[idx_test[-1]+1:]))
    pickle.dump(d, open(f"{save_path}/ind.cora.allx", "wb"))
    pickle.dump(features[idx_test], open(f"{save_path}/ind.cora.tx", "wb"))

    pickle.dump(labels[idx_train], open(f"{save_path}/ind.cora.y", "wb"))
    pickle.dump(labels[idx_test], open(f"{save_path}/ind.cora.ty", "wb"))
    pickle.dump(np.vstack((labels[:idx_test[0]], labels[idx_test[-1]+1:])),
                open(f"{save_path}/ind.cora.ally", "wb"))

    with open(f'{save_path}/ind.cora.test.index', 'w') as f:
        for item in list(idx_test):
            f.write("%s\n" % item)

    # ori_graph
    from collections import defaultdict
    array_adj = np.argwhere(adj.toarray())
    ori_graph = defaultdict(list)
    for edge in array_adj:
        ori_graph[edge[0]].append(edge[1])
    pickle.dump(ori_graph, open(f"{save_path}/ind.cora.graph", "wb"))

    return features, labels, adj


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def check_files(features, labels, adj, dataset):
    dataset = 'cora'
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/cora_nonorm/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "./data/cora_nonorm/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    p_features = sp.vstack(
        (allx[:test_idx_range[0]], tx, allx[test_idx_range[0]:])).tolil()
    p_features[test_idx_reorder, :] = features[test_idx_range, :]

    o_adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    o_labels = np.vstack((ally[:test_idx_range[0]], ty, ally[test_idx_range[0]:]))
    o_labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    assert (p_features.nonzero()[0] == features.nonzero()[0]).all()
    assert (p_features.nonzero()[1] == features.nonzero()[1]).all()
    assert (adj.shape == o_adj.shape)
    print('all files are checked....')


if __name__ == '__main__':
    path = "./data/cora_ori/cora/"
    dataset = "cora"
    save_path = "./data/cora_ori"
    feats, labels, adj = get_features_and_labels(path, dataset, save_path)
    check_files(feats, labels, adj, dataset)
