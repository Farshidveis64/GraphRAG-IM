#!/usr/bin/env python3
"""Run GraphRAG-IM inference and evaluate the selected seeds.

Loads a dataset, runs the full inference engine (retrieve -> encode -> score ->
select), and reports influence spread plus seed-diversity metrics under the
content-conditioned IC model. With --compare_selection it also evaluates the
top-k baseline using common random numbers. Writes results.json and LaTeX tables.

Examples
--------
    python scripts/evaluate.py --config configs/default.yaml --dataset synthetic \\
        --k 20 --compare_selection --output outputs
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_dataset  # noqa: E402
from inference.diffusion import ContentICModel  # noqa: E402
from inference.inference import GraphRAGIMInference  # noqa: E402
from inference.selection import topk_select  # noqa: E402
from models.model_utils import make_rng, seed_everything  # noqa: E402
from utils.config import GraphRAGIMConfig  # noqa: E402
from utils.metrics import (MetricsTracker, detect_communities,  # noqa: E402
                           metrics_to_latex, seeds_to_latex)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
LOG = logging.getLogger("evaluate")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--scoring", default="heuristic", choices=["heuristic", "neural"])
    parser.add_argument("--summarizer", default="offline", choices=["offline", "openai"])
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--eval_sims", type=int, default=300)
    parser.add_argument("--compare_selection", action="store_true")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = GraphRAGIMConfig.from_yaml(args.config) if args.config else GraphRAGIMConfig()
    cfg.seed = args.seed
    cfg.data.name = args.dataset
    if args.k is not None:
        cfg.selection.k = args.k
    os.makedirs(args.output, exist_ok=True)
    rng = make_rng(args.seed)

    synth_kwargs = (dict(
        n_communities=cfg.data.n_communities,
        nodes_per_community=cfg.data.nodes_per_community,
        p_in=cfg.data.p_in, p_out=cfg.data.p_out,
        embedding_dim=cfg.data.embedding_dim,
        topic_separation=cfg.data.topic_separation, seed=cfg.seed)
        if args.dataset == "synthetic" else {})
    cg = load_dataset(args.dataset, root=cfg.data.root, **synth_kwargs)
    cg.ensure_embeddings(dim=cfg.data.embedding_dim, seed=cfg.seed)
    LOG.info("Graph: %d nodes, %d edges", cg.n_nodes, cg.n_edges)

    engine = GraphRAGIMInference(cfg, logger=LOG)
    result = engine.run(cg, scoring=args.scoring, summarizer=args.summarizer, rng=rng)

    model = ContentICModel(cg.graph, cg.embeddings, p0=cfg.diffusion.p0,
                           rho_mode=cfg.diffusion.rho_mode,
                           rho_temperature=cfg.diffusion.rho_temperature)
    labels = detect_communities(cg.graph, cg.communities)
    tracker = MetricsTracker(model, result.text_embeddings, labels)

    dpp = tracker.add("GraphRAG-IM", result.seeds, n_sims=args.eval_sims,
                      rng=np.random.default_rng(args.seed + 1))
    LOG.info("GraphRAG-IM (DPP): %s", {k: round(v, 3) for k, v in dpp.items()})
    summary = {"method": "GraphRAG-IM", "k": cfg.selection.k, **dpp,
               "seeds": [int(s) for s in result.seeds]}

    if args.compare_selection:
        topk = topk_select(result.scores, cfg.selection.k)
        tk = tracker.add("top-k", topk, n_sims=args.eval_sims,
                         rng=np.random.default_rng(args.seed + 1))
        LOG.info("top-k baseline: %s", {k: round(v, 3) for k, v in tk.items()})
        summary["topk"] = tk

    with open(os.path.join(args.output, "results.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(args.output, "seeds_table.tex"), "w") as fh:
        fh.write(seeds_to_latex(result.seeds, caption="GraphRAG-IM selected seeds"))
    with open(os.path.join(args.output, "results_table.tex"), "w") as fh:
        fh.write(metrics_to_latex(tracker.as_dict(), caption="Influence + diversity"))
    cfg.to_yaml(os.path.join(args.output, "config_used.yaml"))
    LOG.info("Wrote results + LaTeX tables to %s/", args.output)


if __name__ == "__main__":
    main()
