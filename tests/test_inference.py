import os
import subprocess
import sys

import numpy as np

from data.dataset import generate_synthetic_graph
from inference.diffusion import ContentICModel
from inference.inference import GraphRAGIMInference
from models.model_utils import OfflineTextEncoder, make_rng
from training.labels import generate_influence_labels
from utils.config import GraphRAGIMConfig


def _small_config():
    cfg = GraphRAGIMConfig()
    cfg.data.n_communities = 3
    cfg.data.nodes_per_community = 20
    cfg.retrieval.max_nodes = 6
    cfg.selection.k = 5
    return cfg


def test_end_to_end_runs():
    cfg = _small_config()
    cg = generate_synthetic_graph(n_communities=3, nodes_per_community=20, seed=0)
    result = GraphRAGIMInference(cfg).run(cg, rng=make_rng(0))
    assert len(result.seeds) == 5
    assert len(set(result.seeds)) == 5
    assert result.scores.shape[0] == cg.n_nodes
    assert len(result.summaries) == cg.n_nodes


def test_inference_reproducible():
    cfg = _small_config()
    cg = generate_synthetic_graph(n_communities=3, nodes_per_community=20, seed=0)
    a = GraphRAGIMInference(cfg).run(cg, rng=make_rng(0))
    b = GraphRAGIMInference(cfg).run(cg, rng=make_rng(0))
    assert a.seeds == b.seeds
    assert a.checksums == b.checksums


def test_labels_have_positives_and_prefer_central():
    cg = generate_synthetic_graph(n_communities=3, nodes_per_community=20, seed=1)
    model = ContentICModel(cg.graph, cg.embeddings, p0=0.2, rho_mode="cosine")
    labels = generate_influence_labels(model, k=5, n_configs=60, n_sims=200,
                                       rng=np.random.default_rng(0))
    assert labels.sum() >= 1
    deg = np.array([d for _, d in cg.graph.degree()])
    assert deg[labels == 1].mean() >= deg.mean() - 1e-9


def test_offline_encoder_stable_across_processes():
    # BLAKE2b (not Python's salted hash) => identical embeddings across processes
    ref = OfflineTextEncoder(dim=32, seed=7).encode(["graph network", "alpha beta"])
    code = (
        "import numpy as np, sys, os;"
        "sys.path.insert(0, os.getcwd());"
        "from models.model_utils import OfflineTextEncoder;"
        "e=OfflineTextEncoder(dim=32, seed=7).encode(['graph network','alpha beta']);"
        "print(float(np.abs(e).sum()))"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONHASHSEED="0")
    out = subprocess.check_output([sys.executable, "-c", code], cwd=root, env=env)
    assert abs(float(out.strip()) - float(np.abs(ref).sum())) < 1e-9
