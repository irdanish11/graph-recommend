from typing import List
import pandas as pd
import torch
import torch.nn.functional as F


def compute_loss(pos_score: torch.Tensor, neg_score: torch.Tensor
                 ) -> torch.Tensor:
    """
    Computes cross entropy loss on edge features

    Source: https://docs.dgl.ai/en/latest/new-tutorial/4_link_predict.html
    """
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def get_hits(edges_df: pd.DataFrame, h: torch.Tensor, h_test: torch.Tensor
             ) -> List[int]:
    """
    Gets list of hits given the parameters below

    parameters: edges list df,
                h embeddings from training,
                h_test embeddings from test set.
    returns: list of hits
    """
    hits = []
    edges = edges_df
    for i in range(h.shape[0]):
        true_edges = list(edges[edges.asin == i].also_bought)
        dist = torch.cdist(h_test[[i]], h)
        top_k = torch.topk(dist, k=500, largest=False)[1]
        hit = 0
        for j in true_edges:
            if j in top_k:
                hit = 1
                break
        hits.append(hit)
    return hits
