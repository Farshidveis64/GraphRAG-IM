from typing import Iterable, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


class ContentICModel:
    """Monte-Carlo content-conditioned Independent Cascade simulator.

    Edge activation probability is p_uv = p0 * rho(x_u, x_v), where rho is a
    content engagement factor. rho_mode='uniform' recovers the classical
    content-blind IC model (rho == 1); 'cosine' derives engagement from the
    cosine similarity of node embeddings. The same model is used for label
    generation and for evaluation, so the diffusion process is never ambiguous.
    """

    def __init__(self, graph: nx.Graph, embeddings: Optional[np.ndarray] = None,
                 p0: float = 0.1, rho_mode: str = "uniform",
                 rho_temperature: float = 1.0, directed: bool = False):
        if rho_mode not in {"uniform", "cosine"}:
            raise ValueError("rho_mode must be 'uniform' or 'cosine'")
        if rho_mode == "cosine" and embeddings is None:
            raise ValueError("cosine rho_mode requires node embeddings")

        self.n = graph.number_of_nodes()
        self.p0 = float(p0)
        self.rho_mode = rho_mode
        self.rho_temperature = float(rho_temperature)
        self.directed = directed

        self._embeddings = None
        if embeddings is not None:
            emb = np.asarray(embeddings, dtype=np.float64)
            self._embeddings = emb / np.clip(
                np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)

        self._build_csr(graph)

    def _build_csr(self, graph: nx.Graph) -> None:
        rows, cols = [], []
        for u, v in graph.edges():
            rows.append(u)
            cols.append(v)
            if not graph.is_directed():
                rows.append(v)
                cols.append(u)
        order = np.lexsort((cols, rows)) if rows else np.array([], dtype=int)
        rows = np.asarray(rows, dtype=np.int64)[order]
        cols = np.asarray(cols, dtype=np.int64)[order]

        indptr = np.zeros(self.n + 1, dtype=np.int64)
        np.add.at(indptr, rows + 1, 1)
        np.cumsum(indptr, out=indptr)

        self.indptr = indptr
        self.indices = cols
        self.probs = self._edge_probabilities(rows, cols)

    def _edge_probabilities(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        if self.rho_mode == "uniform" or src.size == 0:
            return np.full(src.shape, self.p0, dtype=np.float64)
        cos = np.sum(self._embeddings[src] * self._embeddings[dst], axis=1)
        rho = np.clip(np.power((cos + 1.0) / 2.0, self.rho_temperature), 1e-6, 1.0)
        return self.p0 * rho

    def _single_cascade(self, seeds: Sequence[int],
                        rng: np.random.Generator) -> np.ndarray:
        active = np.zeros(self.n, dtype=bool)
        active[list(seeds)] = True
        frontier = list(seeds)
        while frontier:
            new_frontier = []
            for u in frontier:
                start, end = self.indptr[u], self.indptr[u + 1]
                if start == end:
                    continue
                nbrs = self.indices[start:end]
                fired = rng.random(nbrs.shape[0]) < self.probs[start:end]
                for v in nbrs[fired]:
                    if not active[v]:
                        active[v] = True
                        new_frontier.append(int(v))
            frontier = new_frontier
        return active

    def simulate(self, seeds: Iterable[int], n_sims: int = 1000,
                 rng: Optional[np.random.Generator] = None,
                 return_frequency: bool = False) -> Tuple[float, Optional[np.ndarray]]:
        """Estimate expected spread sigma(S) by Monte Carlo."""
        if rng is None:
            rng = np.random.default_rng(0)
        seeds = list(seeds)
        total = 0
        freq = np.zeros(self.n, dtype=np.float64) if return_frequency else None
        for _ in range(n_sims):
            active = self._single_cascade(seeds, rng)
            total += int(active.sum())
            if freq is not None:
                freq += active
        spread = total / n_sims
        if freq is not None:
            freq /= n_sims
        return spread, freq

    def edge_probability(self, u: int, v: int) -> float:
        start, end = self.indptr[u], self.indptr[u + 1]
        nbrs = self.indices[start:end]
        hit = np.where(nbrs == v)[0]
        return float(self.probs[start:end][hit[0]]) if hit.size else 0.0
