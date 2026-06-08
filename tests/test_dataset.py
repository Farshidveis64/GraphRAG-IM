import os
import tempfile

import networkx as nx
import numpy as np

from data.dataset import (ContentGraph, generate_synthetic_graph, load_dataset,
                          write_content_graph)


def test_synthetic_shapes_and_consistency():
    cg = generate_synthetic_graph(n_communities=4, nodes_per_community=20, seed=0)
    cg.validate()
    assert cg.n_nodes == 80
    assert cg.embeddings.shape == (80, 64)
    assert set(np.unique(cg.communities)) == {0, 1, 2, 3}


def test_synthetic_connected_and_separable():
    cg = generate_synthetic_graph(n_communities=3, nodes_per_community=40,
                                  topic_separation=2.0, seed=2)
    assert nx.is_connected(cg.graph)
    emb, comm = cg.embeddings, cg.communities
    centroids = np.stack([emb[comm == c].mean(0) for c in range(3)])
    within = np.mean([np.linalg.norm(emb[i] - centroids[comm[i]])
                      for i in range(cg.n_nodes)])
    across = np.mean([np.linalg.norm(centroids[a] - centroids[b])
                      for a in range(3) for b in range(3) if a != b])
    assert across > within


def test_synthetic_reproducible():
    a = generate_synthetic_graph(seed=7)
    b = generate_synthetic_graph(seed=7)
    assert np.allclose(a.embeddings, b.embeddings)
    assert a.texts == b.texts


def test_write_and_load_roundtrip():
    cg = generate_synthetic_graph(n_communities=3, nodes_per_community=15, seed=0)
    tmp = tempfile.mkdtemp()
    write_content_graph(cg, os.path.join(tmp, "mydata"))
    loaded = load_dataset("mydata", root=tmp)
    assert loaded.n_nodes == cg.n_nodes
    assert loaded.n_edges == cg.n_edges
    assert np.array_equal(loaded.communities, cg.communities)
    assert loaded.embeddings is None  # disk datasets carry no embeddings


def test_ensure_embeddings_bootstrap_and_idempotent():
    cg = generate_synthetic_graph(n_communities=2, nodes_per_community=10, seed=1)
    original = cg.embeddings.copy()
    cg.ensure_embeddings(dim=999)               # no-op: already present
    assert np.array_equal(cg.embeddings, original)

    cg2 = ContentGraph(graph=cg.graph, texts=cg.texts)
    cg2.ensure_embeddings(dim=32, seed=1)
    assert cg2.embeddings.shape == (cg2.n_nodes, 32)


def test_missing_dataset_raises():
    tmp = tempfile.mkdtemp()
    try:
        load_dataset("nope", root=tmp)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "not found" in str(exc).lower()
