from __future__ import annotations
import gzip
from typing import Iterator, Tuple
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import dgl

from preprocessing.cleaning import DGraph


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


def write_text(path: str, text: str):
    with open(path, 'w') as f:
        f.write(text)


def build_graph(dgraph: DGraph):
    node_features, node_labels, edges_src, edges_dst = dgraph
    g = dgl.graph((edges_src, edges_dst))
    g.ndata['feat'] = node_features
    g.ndata['label'] = node_labels
    return g
