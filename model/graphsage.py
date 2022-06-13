from dgl.nn.pytorch.conv import SAGEConv
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class GraphSAGE(nn.Module):
    """
    DGL 2-Layer GraphSAGE Model
    Source: https://docs.dgl.ai/en/0.4.x/_modules/dgl/nn/pytorch/conv/sageconv.html
    """

    def __init__(self, layers, aggregator):
        super(GraphSAGE, self).__init__()
        sage_layers = []
        for i in range(len(layers) - 1):
            sage_layers.append(SAGEConv(layers[i], layers[i+1], aggregator))
        self.sage_conv = nn.ModuleList(sage_layers)

    def forward(self, g, in_feat):
        h = in_feat
        for SConv in self.sage_conv:
            h = F.relu(SConv(g, h))
        return h


class DotPredictor(nn.Module):
    """
    Dot Predictor of Edges,

    Source:
    https://docs.dgl.ai/en/latest/new-tutorial/4_link_predict.html
    """
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product
            # between the source node feature 'h' and destination node
            # feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need
            # to squeeze it.
            return g.edata['score'][:, 0]
