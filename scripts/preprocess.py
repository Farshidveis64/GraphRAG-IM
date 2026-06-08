#!/usr/bin/env python3
"""Build and export a content-graph dataset to the on-disk format.

Generates a synthetic content graph (or any ContentGraph) and writes it as
edges.txt / texts.txt / communities.txt under <output_dir>/<dataset>/, the
layout that load_dataset(name, root) consumes. Real datasets (Cora-ML, Weibo,
DBLP) should be placed in that same format; nothing is downloaded silently.

Examples
--------
    python scripts/preprocess.py --dataset example --output_dir data/processed
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import generate_synthetic_graph, write_content_graph  # noqa: E402
from utils.config import GraphRAGIMConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="example",
                        help="name of the dataset directory to create")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n_communities", type=int, default=5)
    parser.add_argument("--nodes_per_community", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = GraphRAGIMConfig.from_yaml(args.config) if args.config else GraphRAGIMConfig()

    cg = generate_synthetic_graph(
        n_communities=args.n_communities,
        nodes_per_community=args.nodes_per_community,
        embedding_dim=cfg.data.embedding_dim,
        topic_separation=cfg.data.topic_separation, seed=args.seed)

    out_dir = os.path.join(args.output_dir, args.dataset)
    write_content_graph(cg, out_dir)
    print(f"Wrote {cg.n_nodes} nodes / {cg.n_edges} edges to {out_dir}/")
    print(f"Load with: load_dataset('{args.dataset}', root='{args.output_dir}')")


if __name__ == "__main__":
    main()
