from typing import Dict, Tuple

import networkx as nx
import numpy as np


def structural_features(graph: nx.Graph) -> np.ndarray:
    """Standardized per-node features: degree, PageRank, clustering, betweenness."""
    n = graph.number_of_nodes()
    deg = np.array([d for _, d in graph.degree()], dtype=np.float64)
    pr = nx.pagerank(graph)
    clustering = nx.clustering(graph)
    k = None if n <= 2000 else min(500, n)
    betw = nx.betweenness_centrality(graph, k=k, seed=0)
    feats = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        feats[i] = [deg[i], pr[i], clustering[i], betw[i]]
    mean, std = feats.mean(0), feats.std(0)
    return (feats - mean) / np.where(std > 0, std, 1.0)


def edge_prob_map(model, graph: nx.Graph) -> Dict[Tuple[int, int], float]:
    """Materialize a {(u, v): p_uv} mapping from a diffusion model's CSR arrays."""
    probs: Dict[Tuple[int, int], float] = {}
    for u in range(model.n):
        start, end = model.indptr[u], model.indptr[u + 1]
        for v, p in zip(model.indices[start:end], model.probs[start:end]):
            probs[(u, int(v))] = float(p)
    return probs
