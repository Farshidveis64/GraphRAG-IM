import numpy as np

from inference.selection import (build_dpp_kernel, dpp_greedy_map, select_seeds,
                                 topk_select)


def _data(n=30, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.random(n)
    emb = rng.normal(size=(n, dim))
    return scores, emb


def test_kernel_psd_and_symmetric():
    scores, emb = _data()
    L = build_dpp_kernel(scores, emb, theta=0.1)
    assert np.allclose(L, L.T)
    eigvals = np.linalg.eigvalsh(L)
    assert eigvals.min() > -1e-8


def test_selection_size_and_uniqueness():
    scores, emb = _data()
    seeds = select_seeds(scores, emb, k=10, method="dpp", theta=0.1)
    assert len(seeds) == len(set(seeds)) == 10


def test_large_theta_recovers_topk():
    # theta -> inf: the content term tends to the identity, so DPP == top-k
    scores, emb = _data(n=20)
    dpp = set(select_seeds(scores, emb, k=8, method="dpp", theta=1000.0))
    topk = set(topk_select(scores, 8))
    assert dpp == topk


def test_small_theta_is_degenerate():
    # theta -> 0: rank-1 kernel, greedy MAP terminates after one representative
    scores, emb = _data(n=20)
    L = build_dpp_kernel(scores, emb, theta=0.0)
    selected = dpp_greedy_map(L, k=8)
    assert len(selected) == 1


def test_dpp_avoids_near_duplicates():
    # three tight clusters: DPP should spread across them rather than pick one
    rng = np.random.default_rng(1)
    centers = np.array([[5, 0], [-5, 0], [0, 5]], dtype=float)
    emb = np.repeat(centers, 6, axis=0) + 0.01 * rng.normal(size=(18, 2))
    scores = np.ones(18)
    seeds = select_seeds(scores, emb, k=3, method="dpp", theta=0.1)
    clusters = {s // 6 for s in seeds}
    assert len(clusters) == 3


def test_greedy_logdet_matches_subkernel():
    scores, emb = _data(n=15)
    L = build_dpp_kernel(scores, emb, theta=0.2)
    selected = dpp_greedy_map(L, k=5)
    sub = L[np.ix_(selected, selected)]
    assert np.linalg.slogdet(sub)[0] > 0  # positive determinant => valid selection
