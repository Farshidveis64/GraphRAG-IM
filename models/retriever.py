import hashlib
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


@dataclass
class RetrievalResult:
    target: int
    nodes: List[int]
    prizes: np.ndarray
    composed_text: str


class InfluencePrizedRetriever:
    """Stage 1: per-node influence-prized neighborhood retrieval via PCST.

    For a target node the L-hop candidates are assigned prizes (content
    relevance + structural prior) and edges costs (resistance to propagation);
    a connected prize-collecting subgraph rooted at the target is extracted.
    The 'greedy' backend grows a tree by best positive net gain (no deps); the
    'pcst_fast' backend wraps the Goemans-Williamson approximation if installed.
    """

    def __init__(self, n_hops: int = 2, alpha: float = 0.5, gamma: float = 1.0,
                 structural_prior: str = "pagerank", max_nodes: int = 16,
                 backend: str = "greedy"):
        self.n_hops = n_hops
        self.alpha = alpha
        self.gamma = gamma
        self.structural_prior = structural_prior
        self.max_nodes = max_nodes
        self.backend = backend

    def retrieve(self, graph: nx.Graph, embeddings: np.ndarray, target: int,
                 edge_prob: Dict[Tuple[int, int], float],
                 texts: Optional[List[str]] = None,
                 rng: Optional[np.random.Generator] = None) -> RetrievalResult:
        if rng is None:
            rng = np.random.default_rng(0)

        candidates = list(nx.single_source_shortest_path_length(
            graph, target, cutoff=self.n_hops).keys())
        prior = self._structural_prior(graph)

        prizes: Dict[int, float] = {}
        for u in candidates:
            sim = _cosine01(embeddings[target], embeddings[u])
            prizes[u] = self.alpha * sim + (1.0 - self.alpha) * prior[u]
        prizes[target] = float(max(prizes.values(), default=1.0) + 1.0)

        if self.backend == "pcst_fast":
            chosen = self._solve_pcst_fast(graph, candidates, prizes, edge_prob, target)
        else:
            chosen = self._solve_greedy(graph, set(candidates), prizes, edge_prob, target)

        chosen = sorted(set(chosen) | {target}, key=lambda u: prizes[u], reverse=True)
        raw = np.array([prizes[u] for u in chosen], dtype=np.float64)
        norm = raw / raw.sum() if raw.sum() > 0 else np.ones_like(raw) / len(raw)
        composed = _compose_text(target, chosen, norm, texts, rng) if texts else ""
        return RetrievalResult(target=target, nodes=chosen, prizes=norm,
                               composed_text=composed)

    def _structural_prior(self, graph: nx.Graph) -> np.ndarray:
        n = graph.number_of_nodes()
        if self.structural_prior == "degree":
            vals = np.array([d for _, d in graph.degree()], dtype=np.float64)
        else:
            pr = nx.pagerank(graph)
            vals = np.array([pr[i] for i in range(n)], dtype=np.float64)
        span = vals.max() - vals.min()
        return (vals - vals.min()) / span if span > 0 else np.zeros(n)

    def _solve_greedy(self, graph, cand_set, prizes, edge_prob, target) -> List[int]:
        tree, spent, budget = {target}, 0.0, self.gamma * self.max_nodes
        while len(tree) < self.max_nodes:
            best_u, best_gain, best_cost = None, 0.0, 0.0
            for a in tree:
                for u in graph.neighbors(a):
                    if u in tree or u not in cand_set:
                        continue
                    p = edge_prob.get((a, u), edge_prob.get((u, a), 0.0))
                    cost = self.gamma * (1.0 - p)
                    gain = prizes[u] - cost
                    if gain > best_gain or (gain == best_gain and
                                            (best_u is None or u < best_u)):
                        best_u, best_gain, best_cost = u, gain, cost
            if best_u is None or best_gain <= 0.0 or spent + best_cost > budget:
                break
            tree.add(best_u)
            spent += best_cost
        return list(tree)

    def _solve_pcst_fast(self, graph, candidates, prizes, edge_prob, target) -> List[int]:
        import pcst_fast

        idx = {u: i for i, u in enumerate(candidates)}
        prize_arr = np.array([prizes[u] for u in candidates], dtype=np.float64)
        edges, costs = [], []
        for u in candidates:
            for w in graph.neighbors(u):
                if w in idx and u < w:
                    p = edge_prob.get((u, w), edge_prob.get((w, u), 0.0))
                    edges.append([idx[u], idx[w]])
                    costs.append(self.gamma * (1.0 - p))
        edges = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), int)
        vertices, _ = pcst_fast.pcst_fast(
            edges, prize_arr, np.array(costs), idx[target], 1, "gw", 0)
        return [candidates[i] for i in vertices]


class Summarizer(ABC):
    @abstractmethod
    def summarize(self, user_text: str, neighbor_text: str) -> str:
        ...


class OfflineExtractiveSummarizer(Summarizer):
    """Deterministic extractive summarizer (no network / API key)."""

    def __init__(self, max_keywords: int = 20):
        self.max_keywords = max_keywords

    def summarize(self, user_text: str, neighbor_text: str) -> str:
        tokens = (user_text + " " + neighbor_text).lower().split()
        counts = Counter(t for t in tokens if t.isalpha() and len(t) > 2)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return "influence profile: " + ", ".join(
            tok for tok, _ in ranked[:self.max_keywords])


class OpenAISummarizer(Summarizer):
    """LLM summarizer (paper default gpt-3.5-turbo-0125, temperature 0)."""

    PROMPT = ("Summarize this user's influence potential in 100 words based on: "
              "(1) expertise/topics, (2) engagement quality, (3) community standing. "
              "User content: [{user}]. Neighbor content: [{neighbors}].")

    def __init__(self, model: str = "gpt-3.5-turbo-0125", temperature: float = 0.0,
                 max_input_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
        self.max_input_tokens = max_input_tokens
        self._client = None

    def summarize(self, user_text: str, neighbor_text: str) -> str:  # pragma: no cover
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        neighbor_text = " ".join(neighbor_text.split()[:self.max_input_tokens])
        prompt = self.PROMPT.format(user=user_text, neighbors=neighbor_text)
        resp = self._client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()


def get_summarizer(name: str = "offline", **kwargs) -> Summarizer:
    name = name.lower()
    if name == "offline":
        return OfflineExtractiveSummarizer(**kwargs)
    if name == "openai":
        return OpenAISummarizer(**kwargs)
    raise ValueError(f"Unknown summarizer {name!r}")


def summary_checksum(summary: str) -> str:
    return hashlib.sha256(summary.encode("utf-8")).hexdigest()[:16]


def _cosine01(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float((np.dot(a, b) / denom + 1.0) / 2.0)


def _compose_text(target, nodes, norm_prizes, texts, rng) -> str:
    parts = [texts[target]]
    for u, p in zip(nodes, norm_prizes):
        if u == target:
            continue
        tokens = texts[u].split()
        if not tokens:
            continue
        keep = rng.random(len(tokens)) < float(p) * len(nodes)
        kept = [t for t, k in zip(tokens, keep) if k]
        if kept:
            parts.append(" ".join(kept))
    return " \n ".join(parts)
