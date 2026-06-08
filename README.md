# GraphRAG-IM: Retrieval-Augmented Graph Neural Networks for Influence Maximization

A content-aware influence-maximization framework that models diffusion as a
joint function of network structure and node content, in three stages:
(1) influence-prized neighborhood **retrieval** (PCST) + summarization,
(2) dual-branch **encoding** (text + structure) fused by cross-attention,
(3) MLP influence scoring + content-conditioned **DPP** seed selection.

The novel algorithms run with only `numpy`/`networkx` (no GPU, no API key); the
neural encoders are PyTorch reference modules behind optional extras.

## Installation

### Requirements

- Python >= 3.10
- numpy, scipy, networkx, scikit-learn, matplotlib, pyyaml (core)
- PyTorch >= 2.1 + torch-geometric + sentence-transformers (optional, neural path)

### Setup

```bash
git clone https://github.com/your-repo/graphrag-im.git
cd graphrag-im
pip install -r requirements.txt
```

The core install runs the full offline pipeline, the tests, and the figures.
Uncomment the optional block in `requirements.txt` (or `pip install torch
torch-geometric sentence-transformers openai`) to enable the neural scorer and
the LLM summarizer.

## Quick Start

### 1. Data Preprocessing

Build an on-disk example dataset (the layout real datasets use):

```bash
python scripts/preprocess.py --dataset example --output_dir data/processed
```

Real datasets (Cora-ML, Weibo, DBLP) are not bundled. Place them under
`data/processed/<name>/` as `edges.txt` (`u v` per line), `texts.txt` (one
node's text per line), and optional `communities.txt` (one label per line).

### 2. Inference & Evaluation

Run the full pipeline offline and compare content-conditioned DPP with top-k:

```bash
python scripts/evaluate.py 

### 3. Training

Train the neural scorer (requires the optional extras):

```bash
python scripts/train.py 
```




### Retrieval (Stage 1)
```yaml
retrieval:
  n_hops: 2
  alpha: 0.5             # content vs. structure prize trade-off
  max_nodes: 16          # PCST budget
  backend: "greedy"      # 'greedy' (no deps) | 'pcst_fast'
  summarizer: "offline"  # 'offline' (deterministic) | 'openai'
```

### Selection (Stage 3)
```yaml
selection:
  k: 50
  method: "dpp"          # 'dpp' (content-conditioned) | 'topk'
  theta: 0.1             # content-dissimilarity strength
```

Note on the DPP kernel `L_ij = s_i s_j exp(-theta ||h_i-h_j||^2)`: plain top-k
is the `theta -> inf` limit (the content term tends to the identity); `theta -> 0`
is the degenerate rank-1 case. Content conditioning strengthens as `theta`
decreases.

## Project Structure

```
graphrag-im/
├── data/                   # Dataset loading and synthetic generation
│   ├── dataset.py         # ContentGraph, synthetic generator, disk loader, exporter
│   └── data_utils.py      # structural features, edge-probability map
├── models/                 # Model implementations
│   ├── unified_model.py   # GraphSAGE + cross-attention + MLP scorer (+ offline scorer)
│   ├── retriever.py       # influence-prized PCST retriever + summarizers
│   └── model_utils.py     # seeding, device helpers, offline text encoder
├── training/               # Training components
│   ├── trainer.py         # label generation + scorer training loop
│   ├── labels.py          # weak influence-label generation
│   └── losses.py          # BCE objective (L2 via weight decay)
├── inference/              # Inference engine
│   ├── inference.py       # end-to-end retrieve -> encode -> score -> select
│   ├── diffusion.py       # content-conditioned Independent Cascade model
│   └── selection.py       # content-conditioned DPP (greedy MAP) + top-k
├── utils/                  # Utilities
│   ├── config.py          # configuration management
│   ├── metrics.py         # spread + diversity metrics, MetricsTracker, LaTeX export
│   └── visualize.py       # publication-ready plotting
├── scripts/                # Executable scripts
│   ├── preprocess.py      # build/export a dataset
│   ├── train.py           # training script
│   ├── evaluate.py        # inference + evaluation
│   └── visualize.py       # regenerate the three figures
├── tests/                  # Unit tests + pytest-free runner
└── configs/                # Configuration files
    └── default.yaml       # Default configuration
```
