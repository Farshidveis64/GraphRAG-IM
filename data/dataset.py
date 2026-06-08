import os
from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx
import numpy as np

# topic-keyworded vocabulary so an offline bag-of-words encoder recovers structure
_TOPIC_WORDS = [
    ["graph", "network", "node", "edge", "spectral", "laplacian", "topology"],
    ["language", "token", "transformer", "attention", "embedding", "semantic"],
    ["vision", "image", "pixel", "convolution", "detection", "segmentation"],
    ["biology", "protein", "gene", "cell", "sequence", "molecular", "enzyme"],
    ["finance", "market", "asset", "risk", "portfolio", "trading", "volatility"],
    ["climate", "carbon", "emission", "ocean", "atmosphere", "temperature"],
    ["robotics", "control", "sensor", "actuator", "policy", "trajectory"],
    ["security", "encryption", "protocol", "adversary", "key", "authentication"],
]


@dataclass
class ContentGraph:
    """A network together with its node-level content."""

    graph: nx.Graph
    texts: List[str]
    embeddings: Optional[np.ndarray] = None
    communities: Optional[np.ndarray] = None
    name: str = "graph"
    structural_features: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def directed(self) -> bool:
        return self.graph.is_directed()

    def validate(self) -> "ContentGraph":
        n = self.n_nodes
        assert set(self.graph.nodes) == set(range(n)), \
            "nodes must be a contiguous 0..N-1 integer range"
        assert len(self.texts) == n, "one text per node is required"
        if self.embeddings is not None:
            assert self.embeddings.shape[0] == n, "embeddings/node count mismatch"
        if self.communities is not None:
            assert len(self.communities) == n, "communities/node count mismatch"
        return self

    def ensure_embeddings(self, dim: int = 64, seed: int = 42) -> "ContentGraph":
        """Derive content embeddings from text when none are present."""
        if self.embeddings is None:
            from models.model_utils import OfflineTextEncoder

            self.embeddings = OfflineTextEncoder(dim=dim, seed=seed).encode(self.texts)
        return self


def generate_synthetic_graph(n_communities: int = 6, nodes_per_community: int = 60,
                             p_in: float = 0.10, p_out: float = 0.004,
                             embedding_dim: int = 64, topic_separation: float = 2.0,
                             seed: int = 42, **_) -> ContentGraph:
    """Generate a stochastic-block-model content graph with topical communities."""
    rng = np.random.default_rng(seed)
    n_communities = min(n_communities, len(_TOPIC_WORDS))
    sizes = [nodes_per_community] * n_communities

    probs = np.full((n_communities, n_communities), p_out)
    np.fill_diagonal(probs, p_in)
    graph = nx.stochastic_block_model(
        sizes, probs.tolist(), seed=int(rng.integers(0, 2 ** 31 - 1)))
    graph = nx.convert_node_labels_to_integers(graph)
    _connect_components(graph)

    n = graph.number_of_nodes()
    communities = np.repeat(np.arange(n_communities), nodes_per_community)[:n]

    # centroid norm scales with sqrt(dim) so topic signal survives isotropic noise
    centroids = rng.normal(size=(n_communities, embedding_dim))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids *= topic_separation * np.sqrt(embedding_dim)
    embeddings = (centroids[communities] +
                  rng.normal(size=(n, embedding_dim))).astype(np.float32)

    texts = []
    for v in range(n):
        words = _TOPIC_WORDS[communities[v]]
        sample = rng.choice(words, size=int(rng.integers(8, 16)), replace=True)
        texts.append(" ".join(sample))

    return ContentGraph(graph=graph, texts=texts, embeddings=embeddings,
                        communities=communities, name="synthetic").validate()


def load_dataset(name: str, root: str = "data/processed", **kwargs) -> ContentGraph:
    """Load a ContentGraph: 'synthetic' (generated), else a disk dataset under root."""
    name = name.lower()
    if name == "synthetic":
        return generate_synthetic_graph(**kwargs)
    return _load_text_graph(name, root)


def write_content_graph(cg: ContentGraph, out_dir: str) -> None:
    """Export a ContentGraph to the on-disk edges/texts/communities format."""
    cg.validate()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "edges.txt"), "w", encoding="utf-8") as fh:
        for u, v in cg.graph.edges():
            fh.write(f"{u} {v}\n")
    with open(os.path.join(out_dir, "texts.txt"), "w", encoding="utf-8") as fh:
        for text in cg.texts:
            fh.write(text.replace("\n", " ").strip() + "\n")
    if cg.communities is not None:
        np.savetxt(os.path.join(out_dir, "communities.txt"),
                   np.asarray(cg.communities, dtype=int), fmt="%d")


def _load_text_graph(name: str, root: str) -> ContentGraph:
    """Load a text-attributed graph from root/<name>/{edges,texts[,communities]}.txt."""
    base = os.path.join(root, name)
    edge_path = os.path.join(base, "edges.txt")
    text_path = os.path.join(base, "texts.txt")
    if not (os.path.exists(edge_path) and os.path.exists(text_path)):
        raise FileNotFoundError(
            f"Dataset '{name}' not found under '{base}'. Provide 'edges.txt' and "
            "'texts.txt' (see README > Data), or use name='synthetic'.")

    with open(text_path, "r", encoding="utf-8") as fh:
        texts = [line.rstrip("\n") for line in fh]
    n = len(texts)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    with open(edge_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                graph.add_edge(int(parts[0]), int(parts[1]))

    communities = None
    comm_path = os.path.join(base, "communities.txt")
    if os.path.exists(comm_path):
        communities = np.loadtxt(comm_path, dtype=int)

    return ContentGraph(graph=graph, texts=texts, communities=communities,
                        name=name).validate()


def _connect_components(graph: nx.Graph) -> None:
    components = list(nx.connected_components(graph))
    for comp in components[1:]:
        graph.add_edge(next(iter(components[0])), next(iter(comp)))
        components[0] |= comp
