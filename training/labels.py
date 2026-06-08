from typing import Optional

import numpy as np


def generate_influence_labels(model, k: int, n_configs: int = 100,
                              n_sims: int = 1000, top_pct: float = 0.10,
                              membership: float = 0.50,
                              candidate_pool: Optional[np.ndarray] = None,
                              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Binary influence labels via top-spread membership.

    A node is positive if it appears in at least ``membership`` of the top
    ``top_pct`` highest-spread random seed-set configurations, scored under the
    same ContentICModel used for evaluation.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pool = np.arange(model.n) if candidate_pool is None else np.asarray(candidate_pool)

    configs = np.empty((n_configs, k), dtype=np.int64)
    spreads = np.empty(n_configs, dtype=np.float64)
    for i in range(n_configs):
        seeds = rng.choice(pool, size=k, replace=False)
        configs[i] = seeds
        spreads[i], _ = model.simulate(seeds, n_sims=n_sims, rng=rng)

    n_top = max(1, int(round(top_pct * n_configs)))
    top_idx = np.argsort(spreads)[::-1][:n_top]

    appear = np.zeros(model.n, dtype=np.float64)
    for idx in top_idx:
        appear[configs[idx]] += 1.0

    labels = (appear >= membership * n_top).astype(np.int64)
    if labels.sum() == 0 and appear.max() > 0:
        labels[np.argmax(appear)] = 1
    return labels
