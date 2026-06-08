from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import networkx as nx
import numpy as np


def pairwise_embedding_distance(embeddings: np.ndarray,
                                seeds: Sequence[int]) -> float:
    """Mean pairwise Euclidean distance among seed embeddings (higher = more diverse)."""
    seeds = list(seeds)
    if len(seeds) < 2:
        return 0.0
    pts = np.asarray(embeddings)[seeds]
    dists = [np.linalg.norm(pts[i] - pts[j])
             for i, j in combinations(range(len(seeds)), 2)]
    return float(np.mean(dists))


def detect_communities(graph: nx.Graph, ground_truth: Optional[np.ndarray] = None,
                       seed: int = 42) -> np.ndarray:
    """Per-node community label (ground truth if given, else Louvain)."""
    if ground_truth is not None:
        return np.asarray(ground_truth)
    n = graph.number_of_nodes()
    try:
        communities = nx.community.louvain_communities(graph, seed=seed)
    except AttributeError:
        communities = nx.community.greedy_modularity_communities(graph)
    labels = np.full(n, -1, dtype=int)
    for cid, comm in enumerate(communities):
        for node in comm:
            labels[node] = cid
    return labels


def community_coverage(labels: np.ndarray, seeds: Sequence[int]) -> int:
    """Number of distinct communities covered by the seeds (higher = better)."""
    return int(len(set(int(labels[s]) for s in seeds)))


def cascade_overlap(model, seeds: Sequence[int], n_sims: int = 200,
                    threshold: float = 0.1,
                    rng: Optional[np.random.Generator] = None) -> float:
    """Mean pairwise Jaccard overlap of per-seed activation sets (lower = better).

    ``model`` is any diffusion model exposing ``simulate(seeds, n_sims, rng,
    return_frequency=True)`` (e.g. inference.diffusion.ContentICModel).
    """
    seeds = list(seeds)
    if len(seeds) < 2:
        return 0.0
    if rng is None:
        rng = np.random.default_rng(0)
    activation_sets: List[set] = []
    for s in seeds:
        _, freq = model.simulate([s], n_sims=n_sims, rng=rng, return_frequency=True)
        activation_sets.append(set(np.where(freq >= threshold)[0].tolist()))
    overlaps = []
    for a, b in combinations(activation_sets, 2):
        union = a | b
        overlaps.append(len(a & b) / len(union) if union else 0.0)
    return float(np.mean(overlaps))


def influence_spread(model, seeds: Sequence[int], n_sims: int = 300,
                     rng: Optional[np.random.Generator] = None) -> float:
    """Expected influence spread sigma(S) under the given diffusion model."""
    if rng is None:
        rng = np.random.default_rng(0)
    spread, _ = model.simulate(list(seeds), n_sims=n_sims, rng=rng)
    return spread


class MetricsTracker:
    """Accumulates seed sets and reports spread + diversity for each."""

    def __init__(self, model, embeddings: np.ndarray, labels: np.ndarray):
        self.model = model
        self.embeddings = embeddings
        self.labels = labels
        self.records: Dict[str, Dict[str, float]] = {}

    def add(self, name: str, seeds: Sequence[int], n_sims: int = 300,
            rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        # common random numbers: a fresh, identically seeded generator per call
        base = 0 if rng is None else int(rng.integers(0, 2 ** 31 - 1))
        rec = {
            "spread": influence_spread(self.model, seeds, n_sims,
                                       np.random.default_rng(base)),
            "emb_distance": pairwise_embedding_distance(self.embeddings, seeds),
            "community_coverage": community_coverage(self.labels, seeds),
            "cascade_overlap": cascade_overlap(self.model, seeds,
                                               min(n_sims, 200),
                                               rng=np.random.default_rng(base)),
        }
        self.records[name] = rec
        return rec

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return self.records


def seeds_to_latex(seeds: Sequence[int], caption: str = "Selected seed set",
                   label: str = "tab:seeds", per_row: int = 10) -> str:
    """Render a selected seed set as a compact LaTeX table."""
    seeds = list(seeds)
    rows = [" & ".join(str(s) for s in seeds[i:i + per_row])
            for i in range(0, len(seeds), per_row)]
    body = " \\\\\n".join(rows)
    cols = "c" * min(per_row, max(len(seeds), 1))
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
        f"{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def metrics_to_latex(metrics: Mapping[str, Mapping[str, float]],
                     row_order: Iterable[str] = None,
                     caption: str = "Results", label: str = "tab:results",
                     fmt: str = "{:.2f}") -> str:
    """Render a ``{method: {column: value}}`` mapping as a booktabs table."""
    methods = list(row_order) if row_order is not None else list(metrics)
    columns = list(next(iter(metrics.values())).keys())
    header = "Method & " + " & ".join(columns) + " \\\\"
    lines = [
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(columns) + "}", "\\toprule",
        header, "\\midrule",
    ]
    for method in methods:
        cells = [fmt.format(metrics[method][c]) for c in columns]
        lines.append(method + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"
