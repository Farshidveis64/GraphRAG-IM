#!/usr/bin/env python3
"""Regenerate the three publication figures from measured synthetic runs.

Every value is computed by running the library's selection + diffusion code
(nothing hard-coded). Methods are genuine pipeline ablations; spread is scored
with common random numbers. Outputs PDF + PNG to <output_dir>.

Example
-------
    python scripts/visualize.py --output_dir outputs/figures
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import generate_synthetic_graph  # noqa: E402
from inference.diffusion import ContentICModel  # noqa: E402
from inference.selection import select_seeds, topk_select  # noqa: E402
from models.model_utils import OfflineTextEncoder, seed_everything  # noqa: E402
from models.unified_model import HeuristicScorer  # noqa: E402
from utils.metrics import (cascade_overlap, community_coverage,  # noqa: E402
                           detect_communities, pairwise_embedding_distance)
from utils.visualize import plot_ablation, plot_diversity, plot_sensitivity  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
LOG = logging.getLogger("visualize")
EVAL_SEED = 20240608


def _spread(model, seeds, sims=200):
    return model.simulate(seeds, sims, np.random.default_rng(EVAL_SEED))[0]


def _diversity(model, emb, labels, seeds, sims=200):
    return {
        "emb_distance": pairwise_embedding_distance(emb, seeds),
        "community_coverage": community_coverage(labels, seeds),
        "cascade_overlap": cascade_overlap(model, seeds, sims,
                                           rng=np.random.default_rng(EVAL_SEED)),
    }


def _content_scores(cg, seed):
    emb = OfflineTextEncoder(dim=64, seed=seed).encode(cg.texts)
    return HeuristicScorer(beta=0.6).score(cg.graph, emb), emb


def diversity_results(cg, model, emb, labels, k):
    s_struct = HeuristicScorer(beta=1.0).score(cg.graph, cg.embeddings)
    s_content, _ = _content_scores(cg, seed=42)
    seeds = {
        "ToupleGDD": topk_select(s_struct, k),
        "ToupleGDD+Text": topk_select(s_content, k),
        "GraphRAG-IM": select_seeds(s_content, emb, k=k, method="dpp", theta=0.1),
    }
    return {name: _diversity(model, emb, labels, sset) for name, sset in seeds.items()}


def sensitivity_results(cg, emb):
    s_content, _ = _content_scores(cg, seed=42)
    k_values = [5, 10, 20, 30, 40, 50]
    p_values = [0.05, 0.10, 0.15, 0.20]
    spread_by_k = {"ToupleGDD+Text": [], "GraphRAG-IM": []}
    model = ContentICModel(cg.graph, cg.embeddings, p0=0.1, rho_mode="cosine")
    for k in k_values:
        spread_by_k["ToupleGDD+Text"].append(_spread(model, topk_select(s_content, k)))
        spread_by_k["GraphRAG-IM"].append(
            _spread(model, select_seeds(s_content, emb, k, "dpp", 0.1)))
    spread_by_p = {"ToupleGDD+Text": [], "GraphRAG-IM": []}
    for p in p_values:
        m = ContentICModel(cg.graph, cg.embeddings, p0=p, rho_mode="cosine")
        spread_by_p["ToupleGDD+Text"].append(_spread(m, topk_select(s_content, 50)))
        spread_by_p["GraphRAG-IM"].append(
            _spread(m, select_seeds(s_content, emb, 50, "dpp", 0.1)))
    return k_values, spread_by_k, p_values, spread_by_p


def ablation_results(n_graphs=6):
    comps = {"-Retrieval": [], "-Content": [], "-DPP": []}
    k = 30
    for seed in range(1, n_graphs + 1):
        cg = generate_synthetic_graph(n_communities=6, nodes_per_community=60,
                                      topic_separation=2.0, seed=seed)
        model = ContentICModel(cg.graph, cg.embeddings, p0=0.15, rho_mode="cosine")
        s_content, emb = _content_scores(cg, seed=seed)
        s_struct = HeuristicScorer(beta=1.0).score(cg.graph, cg.embeddings)
        s_raw = HeuristicScorer(beta=0.6).score(cg.graph, cg.embeddings)
        full = _spread(model, select_seeds(s_content, emb, k, "dpp", 0.1))

        def drop(seeds, base=full):
            return 100.0 * (base - _spread(model, seeds)) / base

        comps["-Retrieval"].append(
            drop(select_seeds(s_raw, cg.embeddings, k, "dpp", 0.1)))
        comps["-Content"].append(drop(topk_select(s_struct, k)))
        comps["-DPP"].append(drop(topk_select(s_content, k)))
    return {c: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for c, v in comps.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default="outputs/figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    cg = generate_synthetic_graph(n_communities=6, nodes_per_community=60, seed=args.seed)
    model = ContentICModel(cg.graph, cg.embeddings, p0=0.1, rho_mode="cosine")
    _, emb = _content_scores(cg, seed=42)
    labels = detect_communities(cg.graph, cg.communities)

    LOG.info("Computing diversity comparison ...")
    div = diversity_results(cg, model, emb, labels, k=30)
    plot_diversity(div, out_path=os.path.join(args.output_dir, "fig_diversity.pdf"))

    LOG.info("Computing sensitivity sweeps ...")
    kv, sk, pv, sp = sensitivity_results(cg, emb)
    plot_sensitivity(kv, sk, pv, sp,
                     out_path=os.path.join(args.output_dir, "fig_sensitivity.pdf"))

    LOG.info("Computing ablation study ...")
    abl = ablation_results()
    plot_ablation(abl, out_path=os.path.join(args.output_dir, "fig_ablation.pdf"))

    with open(os.path.join(args.output_dir, "figure_data.json"), "w") as fh:
        json.dump({"diversity": div, "sensitivity_k": {"k": kv, **sk},
                   "sensitivity_p": {"p": pv, **sp}, "ablation": abl}, fh, indent=2)
    LOG.info("Wrote figures (PDF+PNG) + figure_data.json to %s/", args.output_dir)


if __name__ == "__main__":
    main()
