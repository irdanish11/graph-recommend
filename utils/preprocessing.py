from sklearn.feature_extraction.text import TfidfVectorizer
from utils import utils
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import torch
import os


DGraph = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def construct_also_bought(df: pd.DataFrame) -> pd.DataFrame:
    # Only keep rows with "Also Bought" in the related column
    df.related = df.related.apply(
        lambda x: x if 'also_bought' in x.keys() else np.nan)
    df = df.dropna()
    # Create also_bought column
    df['also_bought'] = df.related.apply(lambda x: x['also_bought'])
    # Remove all product IDs (ASINs) outside the dataset
    also_bought = []
    df['also_bought'] = df['also_bought'].apply(
        lambda x: set(df.asin).intersection(set(x)))
    df['also_bought'] = df['also_bought'].apply(
        lambda x: list(x) if len(x) > 0 else np.nan)
    df = df.dropna().reset_index(drop=True)
    # split the also bought into multiple rows
    df = df.explode('also_bought')
    return df


def edge_mapping(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    edges = df[['asin', 'also_bought']]
    # Map String ASINs (Product IDs) to Int IDs
    concat = list(set(np.concatenate([edges.asin, edges.also_bought])))
    asin_map = {v: i for i, v in enumerate(concat)}

    edges.asin = edges.asin.apply(lambda x: asin_map[x])
    edges.also_bought = edges.also_bought.apply(lambda x: asin_map[x])
    edges = edges.reset_index(drop=True)
    return edges, asin_map


def get_node_data(df: pd.DataFrame, asin_map: Dict[str, int]
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    all_asin = list(asin_map.keys())
    # Text Manipulations
    text_df = df[['asin', 'title', 'niche']]
    text_df.asin = text_df.asin.apply(
        lambda x: asin_map[x] if x in asin_map else np.nan)
    text_df = text_df.dropna()
    text_df = text_df.drop_duplicates('asin').reset_index(drop=True)

    # TF-IDF Vectorizer for Title Text Feature
    corpus = text_df.title.tolist()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    node_features = torch.Tensor(X.toarray())
    node_labels = torch.from_numpy(
        text_df['niche'].astype('category').cat.codes.to_numpy())
    return node_features, node_labels


def graph_processing(data_path: str, meta_file: str
                     ) -> Tuple[DGraph, pd.DataFrame]:
    print('\n\nLoading Data...')
    df = utils.read_data(os.path.join(data_path, meta_file))
    print('\n\nPreprocessing Graph...')
    df = df.dropna().reset_index(drop=True)
    # Finds Niche Category of each product
    df['niche'] = df.categories.apply(lambda x: x[0][-1])
    df1 = construct_also_bought(df)
    # edges mapping
    edges, asin_map = edge_mapping(df1)
    node_features, node_labels = get_node_data(df, asin_map)
    # source and destination nodes
    edges_src = torch.from_numpy(edges['asin'].to_numpy())
    edges_dst = torch.from_numpy(edges['also_bought'].to_numpy())
    dgraph = (node_features, node_labels, edges_src, edges_dst)
    print('\n\nGraph processing completed!')
    return dgraph, edges


if __name__ == '__main__':
    DATA_PATH = 'dataset/'
    META_FILE = 'meta_Automotive.json.gz'
    graph_tuple, df_edges = graph_processing(DATA_PATH, META_FILE)
