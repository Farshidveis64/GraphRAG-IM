import networkx as nx
import numpy as np

from inference.diffusion import ContentICModel


def test_spread_bounds():
    model = ContentICModel(nx.path_graph(6), p0=0.5, rho_mode="uniform")
    spread, _ = model.simulate([0], n_sims=500, rng=np.random.default_rng(0))
    assert 1.0 <= spread <= 6.0


def test_p0_monotonicity():
    g = nx.gnp_random_graph(40, 0.1, seed=3)
    low, _ = ContentICModel(g, p0=0.05, rho_mode="uniform").simulate(
        [0, 1, 2], 400, np.random.default_rng(0))
    high, _ = ContentICModel(g, p0=0.5, rho_mode="uniform").simulate(
        [0, 1, 2], 400, np.random.default_rng(0))
    assert high > low


def test_edge_probability_matches_formula():
    g = nx.path_graph(3)
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    model = ContentICModel(g, emb, p0=0.4, rho_mode="cosine")
    assert abs(model.edge_probability(0, 1) - 0.4) < 1e-6     # cos=1 -> rho=1
    assert abs(model.edge_probability(1, 2) - 0.2) < 1e-6     # cos=0 -> rho=0.5


def test_reproducible_with_same_rng():
    g = nx.gnp_random_graph(30, 0.1, seed=2)
    model = ContentICModel(g, p0=0.3, rho_mode="uniform")
    s1, _ = model.simulate([0], 200, np.random.default_rng(42))
    s2, _ = model.simulate([0], 200, np.random.default_rng(42))
    assert s1 == s2
