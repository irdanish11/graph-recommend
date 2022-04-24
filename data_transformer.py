#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 26/3/22 2:26 PM
# @author: danish

from __future__ import annotations
import gzip
import os
from typing import Iterator, Tuple, Dict
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', type=str, default='dataset',
                    help='Directory which contains input data and output '
                         'will be stored')
parser.add_argument('--input-file', type=str,
                    default='reviews_Automotive_5.json.gz', help='input file')
parser.add_argument('--output-file', type=str, default='automotive_5.net',
                    help='output file')


def parse(path: str) -> Iterator[dict]:
    for line in gzip.open(path, 'rb'):
        yield eval(line)


def read_data(path: str) -> pd.DataFrame:
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def write_text(path: str, text: str):
    with open(path, 'w') as f:
        f.write(text)


def reviewers_products(df: pd.DataFrame) -> Tuple[Dict[str | int, int],
                                                  Dict[str | int, int]]:
    reviewers = df['reviewerID'].unique()
    products = df['asin'].unique()
    num_reviewers = len(reviewers)

    reviewers_nodes = {r: i+1 for i, r in enumerate(reviewers)}
    products_nodes = {p: num_reviewers+i+1 for i, p in enumerate(products)}
    return reviewers_nodes, products_nodes


def build_netfile(df: pd.DataFrame, reviewers_nodes: Dict[str | int, int],
                  products_nodes: Dict[str | int, int]) -> str:
    rev_mapped = np.array(df['reviewerID'].map(reviewers_nodes))
    products_mapped = np.array(df['asin'].map(products_nodes))

    sorted_indices = np.argsort(rev_mapped)
    rev_mapped = rev_mapped[sorted_indices]
    products_mapped = products_mapped[sorted_indices]

    net_file_list = [f'{r} {p}' for r, p in zip(rev_mapped, products_mapped)]
    net_file = '\n'.join(net_file_list)
    return net_file


if __name__ == '__main__':
    args = parser.parse_args()
    input_path = os.path.join(args.dataset_dir, args.input_file)
    # input_path = 'dataset/reviews_Automotive_5.json.gz'
    df_am = read_data(input_path)
    df_am = df_am.dropna()
    # reviewers node and products node mappings
    rev_nodes, pro_nodes = reviewers_products(df_am)
    # build net file
    graph_automotive = build_netfile(df_am, rev_nodes, pro_nodes)
    # write net file
    output_path = os.path.join(args.dataset_dir, args.output_file)
    write_text(output_path, graph_automotive)
