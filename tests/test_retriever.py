import networkx as nx
import numpy as np

from data.data_utils import edge_prob_map
from inference.diffusion import ContentICModel
from models.retriever import InfluencePrizedRetriever


def _setup(n=40, seed=0):
    g = nx.gnp_random_graph(n, 0.12, seed=seed)
    if not nx.is_connected(g):
        comps = list(nx.connected_components(g))
        for c in comps[1:]:
            g.add_edge(next(iter(comps[0])), next(iter(c)))
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, 8))
    model = ContentICModel(g, emb, p0=0.3, rho_mode="cosine")
    return g, emb, edge_prob_map(model, g)


def test_target_included_and_budget_respected():
    g, emb, probs = _setup()
    r = InfluencePrizedRetriever(max_nodes=8)
    res = r.retrieve(g, emb, 0, probs)
    assert res.target == 0 and 0 in res.nodes
    assert len(res.nodes) <= 8


def test_prizes_normalized():
    g, emb, probs = _setup()
    res = InfluencePrizedRetriever(max_nodes=8).retrieve(g, emb, 1, probs)
    assert abs(res.prizes.sum() - 1.0) < 1e-8
    assert np.all(res.prizes >= 0)


def test_nodes_within_l_hops():
    g, emb, probs = _setup()
    res = InfluencePrizedRetriever(n_hops=2, max_nodes=12).retrieve(g, emb, 2, probs)
    reachable = nx.single_source_shortest_path_length(g, 2, cutoff=2)
    assert all(u in reachable for u in res.nodes)


def test_deterministic():
    g, emb, probs = _setup()
    r = InfluencePrizedRetriever(max_nodes=8)
    a = r.retrieve(g, emb, 0, probs, rng=np.random.default_rng(0))
    b = r.retrieve(g, emb, 0, probs, rng=np.random.default_rng(0))
    assert a.nodes == b.nodes


def test_composed_text():
    g, emb, probs = _setup()
    texts = [f"word{i} topic alpha" for i in range(g.number_of_nodes())]
    res = InfluencePrizedRetriever(max_nodes=8).retrieve(
        g, emb, 0, probs, texts=texts, rng=np.random.default_rng(0))
    assert isinstance(res.composed_text, str) and len(res.composed_text) > 0
