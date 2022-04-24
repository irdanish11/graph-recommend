#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 24/4/22 1:20 AM
# @author: danish

from typing import List, Dict, Tuple
import networkx as nx
from networkx.algorithms import bipartite
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Graph Analysis')
parser.add_argument('--input-dir', help='Input file', default='dataset')
parser.add_argument('--edge-list', help='Input file',
                    default='automotive_5.net')


def read_bipartite_graph(path: str) -> nx.Graph:
    print('Reading graph...')
    graph = bipartite.read_edgelist(path)
    return graph


def plot_distribution(values: List[int], title: str, xlabel: str, ylabel: str,
                      filename: str) -> None:
    sns.distplot(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.show()


def degree_distribution(graph: nx.Graph) -> None:
    degree = graph.degree
    rev_nodes, prod_nodes = bipartite.sets(graph)
    reviewers = range(1, len(rev_nodes)+1)
    bound = len(rev_nodes) + len(prod_nodes) + 1
    products = range(len(rev_nodes)+1, bound)
    reviewers_degree = [degree(str(i)) for i in reviewers]
    products_degree = [degree(str(i)) for i in products]

    plot_distribution(reviewers_degree, 'Reviewer Degree Distribution',
                      'Degree', 'Number of Nodes', 'plots/reviewer_degree.png')
    plot_distribution(products_degree, 'Product Degree Distribution',
                      'Degree', 'Number of Nodes', 'plots/product_degree.png')


def analysis(graph: nx.Graph) -> None:
    print('Analyzing graph...')
    # average shortest path length
    avg_path = nx.average_shortest_path_length(graph)
    print('Average shortest path length:', avg_path)
    diameter = nx.diameter(graph)
    print(f'Diameter of the graph: {diameter}')
    # average clustering coefficient
    avg_cc = bipartite.average_clustering(graph)
    print('Average clustering coefficient:', avg_cc)
    # degree distribution
    degree_distribution(graph)
    # closeness centrality
    cc = bipartite.closeness_centrality(graph, graph.nodes())

    # bc = bipartite.betweenness_centrality(graph, graph.nodes())
    # dc = bipartite.degree_centrality(graph, graph.nodes())
    # density = bipartite.density(graph, graph.nodes())
    # redundancy = bipartite.redundancy(graph, graph.nodes)


if __name__ == '__main__':
    args = parser.parse_args()
    file_path = os.path.join(args.input_dir, args.edge_list)
    # file_path = 'dataset/automotive_5.net'
    bi_graph = read_bipartite_graph(file_path)
    analysis(bi_graph)
