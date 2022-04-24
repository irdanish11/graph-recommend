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


def read_line(path: str) -> str:
    with open(path) as f:
        content = f.readline()
    return content


def read_bipartite_graph(path: str) -> Tuple[Dict[str, int], nx.Graph]:
    print('Reading graph...')
    line = read_line(path)
    node_info = eval(line.replace('#', '').strip())
    graph = bipartite.read_edgelist(path)
    return node_info, graph


def plot_distribution(values: List[int], title: str, xlabel: str, ylabel: str,
                      filename: str) -> None:
    sns.distplot(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.show()


def degree_distribution(node_info: Dict[str, int], graph: nx.Graph) -> None:
    degree = graph.degree
    reviewers = range(1, node_info['reviewers']+1)
    bound = node_info['reviewers']+node_info['products']
    products = range(node_info['reviewers']+3, bound+3)
    reviewers_degree = [degree(str(i)) for i in reviewers]
    products_degree = [degree(str(i)) for i in products]

    plot_distribution(reviewers_degree, 'Reviewer Degree Distribution',
                      'Degree', 'Number of Nodes', 'plots/reviewer_degree.png')
    plot_distribution(products_degree, 'Product Degree Distribution',
                      'Degree', 'Number of Nodes', 'plots/product_degree.png')


def analysis(node_info: Dict[str, int], graph: nx.Graph) -> None:
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
    degree_distribution(node_info, graph)


if __name__ == '__main__':
    args = parser.parse_args()
    file_path = os.path.join(args.input_dir, args.edge_list)
    # file_path = 'dataset/automotive_5.net'
    node_information, bi_graph = read_bipartite_graph(file_path)
    analysis(node_information, bi_graph)
