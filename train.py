import os
from typing import List, Dict, Tuple

import torch
import itertools
import numpy as np
from torch.nn import Module
import matplotlib.pyplot as plt
from utils.preprocessing import graph_processing
from utils.utils import build_graph, train_test_split, write_pickle
from model.graphsage import GraphSAGE, DotPredictor
from eval import compute_loss, get_hits


def plot(val: List[float], title: str, ylabel: str):
    os.makedirs('plots', exist_ok=True)
    plt.plot(val)
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(f'plots/{title}.png')
    plt.show()


def train(data_path: str, meta_file: str, epochs: int, artifacts_path: str
          ) -> Tuple[Dict[str, List[float]], Module]:
    os.makedirs(artifacts_path, exist_ok=True)
    # Preprocess
    dgraph, edges = graph_processing(data_path, meta_file)
    node_features, node_labels, edges_src, edges_dst = dgraph

    # Build DGL Graph
    g = build_graph(dgraph)

    # Train-Test Split Graphs
    data = train_test_split(g, test_split=0.1)
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = data

    # Inits
    model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)
    pred = DotPredictor()
    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
    epochs = epochs

    # Training
    history = {'loss': [], 'hits': []}
    for i, epoch in enumerate(range(epochs)):
        # Forward
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        hits = np.mean(get_hits(edges, h, model(test_pos_g, node_features)))
        history['loss'].append(loss.item())
        history['hits'].append(hits)
        print(f'Epoch: {i+1}/{epochs}, Loss: {loss}, Hits: {hits}')
        write_pickle(history, os.path.join(artifacts_path, 'history.pkl'))
        torch.save(model.state_dict(),
                   os.path.join(artifacts_path, 'model.pth'))
    # plot loss
    plot(history['loss'], 'Training Loss', 'loss')
    plot(history['hits'], 'Hits Rate', 'hits')
    return history, model


if __name__ == '__main__':
    DATA_PATH = 'dataset/'
    META_FILE = 'meta_Automotive.json.gz'
    ARTIFACTS_PATH = 'artifacts/'
    # data_path = DATA_PATH
    # meta_file = META_FILE
    train(DATA_PATH, META_FILE, epochs=2, artifacts_path=ARTIFACTS_PATH)
