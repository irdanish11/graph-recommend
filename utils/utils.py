from __future__ import annotations
import gzip
from typing import Iterator, Tuple
import pandas as pd
import torch
from tqdm import tqdm
import dgl
import numpy as np
import scipy.sparse as sp
from pickle import dump, load


DGraph = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def load_pickle(path: str) -> object:
    with open(path, 'rb') as f:
        return load(f)


def write_pickle(obj: object, path: str) -> None:
    with open(path, 'wb') as f:
        dump(obj, f)


def parse(path: str) -> Iterator[dict]:
    for line in gzip.open(path, 'rb'):
        yield eval(line)


def read_data(path: str) -> pd.DataFrame:
    i = 0
    df = {}
    for d in tqdm(parse(path)):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def write_text(path: str, text: str) -> None:
    with open(path, 'w') as f:
        f.write(text)


def build_graph(dgraph: DGraph) -> dgl.DGLGraph:
    node_features, node_labels, edges_src, edges_dst = dgraph
    g = dgl.graph((edges_src, edges_dst))
    g.ndata['feat'] = node_features
    g.ndata['label'] = node_labels
    return g


def train_test_split(g: dgl.DGLGraph, test_split: float
                     ) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph,
                                dgl.DGLGraph, dgl.DGLGraph]:
    u, v = g.edges()

    edge_ids = np.arange(g.number_of_edges())
    edge_ids = np.random.permutation(edge_ids)
    test_size = int(len(edge_ids) * test_split)
    test_pos_u, test_pos_v = u[edge_ids[:test_size]], v[edge_ids[:test_size]]
    train_pos_u, train_pos_v = u[edge_ids[test_size:]], v[edge_ids[test_size:]]

    # Find all negative edges and split them for training and testing
    shape = (g.number_of_nodes(), g.number_of_nodes())
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape)
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_edge_ids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
    test_neg_u = neg_u[neg_edge_ids[:test_size]]
    test_neg_v = neg_v[neg_edge_ids[:test_size]]
    train_neg_u = neg_u[neg_edge_ids[test_size:]]
    train_neg_v = neg_v[neg_edge_ids[test_size:]]

    train_g = dgl.remove_edges(g, edge_ids[:test_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v),
                            num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v),
                            num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v),
                           num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v),
                           num_nodes=g.number_of_nodes())

    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g
