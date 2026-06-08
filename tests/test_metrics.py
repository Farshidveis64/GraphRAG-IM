import networkx as nx
import numpy as np

from inference.diffusion import ContentICModel
from utils.metrics import (cascade_overlap, community_coverage,
                           detect_communities, pairwise_embedding_distance)


def test_pairwise_distance_zero_for_singleton():
    emb = np.random.default_rng(0).normal(size=(10, 4))
    assert pairwise_embedding_distance(emb, [3]) == 0.0


def test_pairwise_distance_increases_with_spread():
    emb = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]])
    close = pairwise_embedding_distance(emb, [0, 1])
    far = pairwise_embedding_distance(emb, [0, 2])
    assert far > close


def test_community_coverage_counts_distinct():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert community_coverage(labels, [0, 2, 4]) == 3
    assert community_coverage(labels, [0, 1]) == 1


def test_detect_communities_uses_ground_truth():
    g = nx.path_graph(5)
    gt = np.array([0, 0, 1, 1, 1])
    assert np.array_equal(detect_communities(g, gt), gt)


def test_cascade_overlap_in_range_and_singleton_zero():
    g = nx.gnp_random_graph(30, 0.12, seed=1)
    emb = np.random.default_rng(1).normal(size=(30, 6))
    model = ContentICModel(g, emb, p0=0.3, rho_mode="cosine")
    ovl = cascade_overlap(model, [0, 5, 10], n_sims=100, rng=np.random.default_rng(0))
    assert 0.0 <= ovl <= 1.0
    assert cascade_overlap(model, [0], n_sims=100) == 0.0
