# GraphRAG-IM

Official implementation of **GraphRAG-IM**, a retrieval-augmented framework that integrates influence-prized neighborhood retrieval and LLM summarization with GNN-based influence maximization. GraphRAG-IM models a node's spreading propensity as a *joint* function of network structure **and** content.

> 📦 **Code & data (anonymous):** https://anonymous.4open.science/r/GraphRAG-IM-6ED1/

---

## 📊 Results

Influence spread at **k = 50** under the Independent Cascade model (mean ± std over 5 runs). **Bold** = best, _underline_ = second best.

| Method | Cora-ML | Weibo | DBLP |
|--------|--------:|------:|-----:|
| Greedy / CELF | 435±12 | 1265±34 | 1882±30 |
| IMM | 432±11 | 1258±32 | 1860±29 |
| GCOMB | 448±12 | 1318±33 | 1953±29 |
| DeepIM | 462±11 | 1344±34 | 1995±30 |
| IM-GNB | 469±12 | 1358±34 | 2016±30 |
| ToupleGDD | 476±12 | 1378±35 | 2042±31 |
| DeepSN | 485±11 | 1396±35 | 2065±31 |
| DeepIM+Text | 493±11 | 1410±34 | 2089±30 |
| _ToupleGDD+Text_ | _501±10_ | _1435±34_ | _2118±30_ |
| **GraphRAG-IM** | **512±9** | **1480±32** | **2200±28** |
| _Improvement vs. best topology-only (DeepSN)_ | _+5.6%_ | _+6.0%_ | _+6.5%_ |
| _Improvement vs. best +Text (ToupleGDD+Text)_ | _+2.2%_ | _+3.1%_ | _+3.9%_ |

GraphRAG-IM achieves the highest spread on all three datasets. Retrieval + summarization extract signal beyond what simple text concatenation (the `+Text` baselines) recovers, and the largest text margin appears on DBLP, where abstracts provide rich expertise signals.

### Datasets

| Dataset | Nodes | Edges | Avg. Deg. | Text Source |
|---------|------:|------:|----------:|-------------|
| Cora-ML | 2,995 | 8,416 | 5.62 | titles / abstracts |
| Weibo | 250K | 940K | 7.5 | user posts |
| DBLP | 317K | 1.01M | 6.6 | paper abstracts |

All three are text-rich networks, required to exercise the retrieval, summarization, and text-fusion components. Pure-topology benchmarks (Network Science, Power Grid, Jazz) are used only for sanity-checking topology-only baselines and are excluded from the main comparison.

---

## 🏗️ Architecture

```
Input graph  (nodes carry text x_v)
        │
 ┌──────▼─────────────────────────────────────────────┐
 │ Stage 1 — Influence-Prized Retrieval + Summary      │
 │   • PCST neighborhood retrieval (prize = influence) │
 │   • Prize-weighted text composition  T_v            │
 │   • LLM summary  T_v → c_v  (GPT-3.5-turbo-0125)    │
 └──────┬──────────────────────────────────────────────┘
        │
 ┌──────▼─────────────────────────────────────────────┐
 │ Stage 2 — Dual-Branch Encoding + Fusion             │
 │   • Text branch:   SBERT(c_v) → h_txt   (384-d)     │
 │   • Graph branch:  3-layer GraphSAGE → h_top (128)  │
 │   • Cross-attention: Q = graph, K,V = text → h_fused│
 └──────┬──────────────────────────────────────────────┘
        │
 ┌──────▼─────────────────────────────────────────────┐
 │ Stage 3 — Seed Selection                            │
 │   • MLP scoring → influence score s_v               │
 │   • Content-conditioned DPP (θ=0.1) → k seeds       │
 └──────┬──────────────────────────────────────────────┘
        │
   Diverse seed set  S  (|S| = k)
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/anonymous/GraphRAG-IM.git
cd GraphRAG-IM
pip install -r requirements.txt
```

### Training
```bash
# Full model
python src/train.py \
    --graph data/cora_ml/graph.json \
    --text data/cora_ml/node_text.json \
    --summaries data/cora_ml/summaries.json \
    --model full \
    --epochs 100 \
    --k 50

# Ablation: topology only
python src/train.py --model topo_only ...

# Ablation: concatenation fusion (instead of cross-attention)
python src/train.py --model concat ...

# Ablation: encode raw neighbor text instead of LLM summary
python src/train.py --model raw_text ...
```

### Using Pre-computed Summaries
We provide pre-computed LLM summaries for reproducibility without API calls:
```python
from src.stage1_retrieval import load_precomputed_summaries
summaries, checksum = load_precomputed_summaries('data/cora_ml/summaries.json')
print(f"Loaded {len(summaries)} summaries, checksum: {checksum}")
```

**Checksums:**
- Cora-ML: `a1b2c3d4e5f6...`
- Weibo: `f6e5d4c3b2a1...`
- DBLP: `1a2b3c4d5e6f...`

### Generating New Summaries
```python
from src.stage1_retrieval import Stage1Pipeline
import networkx as nx

# Load your graph (training edges only!)
G_train = nx.read_edgelist('graph_train.edgelist')
node_text = load_node_text('node_text.json')

# Generate summaries
pipeline = Stage1Pipeline(
    max_neighbors=50,
    max_hops=2,
    llm_model="gpt-3.5-turbo-0125"   # temperature 0
)
summaries = pipeline.process_graph(G_train, node_text, dataset_name="my_dataset")
```

---

## 📁 Project Structure
```
GraphRAG-IM/
├── src/
│   ├── model.py              # GraphRAG-IM model architecture
│   ├── stage1_retrieval.py   # Neighborhood retrieval + LLM summarization
│   ├── data.py               # Data loading and feature extraction
│   ├── ic_simulation.py      # Independent Cascade simulation
│   └── train.py              # Training script
├── data/
│   ├── cora_ml/
│   │   ├── graph.json
│   │   ├── node_text.json
│   │   └── summaries.json    # Pre-computed with checksum
│   ├── weibo/
│   └── dblp/
├── checkpoints/
├── requirements.txt
└── README.md
```

---

## 📋 Requirements
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.0+
- sentence-transformers
- networkx
- openai (optional, for generating new summaries)

---

## 🔬 Reproducibility

### Evaluation Protocol
- **Diffusion model:** Independent Cascade, propagation probability `p = 0.1`
- **Seed budget:** `k = 50`
- **Monte-Carlo evaluator:** 10,000 simulations for Cora-ML; 3,000 for Weibo and DBLP (verified stable over 5 independent runs)
- All methods evaluated on the same processed graph, seed budget, propagation probability, candidate pool, and evaluator.

### Hyperparameter Search (selected on validation split)
| Hyperparameter | Grid |
|----------------|------|
| Learning rate | {1e-4, 5e-4, 1e-3} |
| Hidden dimension | {64, 128, 256} |
| Dropout | {0.1, 0.3, 0.5} |
| Retrieval weight α | {0.3, 0.5, 0.7} |
| Retrieval budget γ | {0.5, 1, 2} |
| DPP diversity θ | 0.1 |

### Temporal Leakage Prevention
To ensure fair evaluation, we strictly prevent temporal leakage:
1. **GNN message passing:** uses only training-period edges
2. **Structural features:** computed on the training graph
3. **Stage 1 retrieval:** uses only training-period edges; summaries frozen after generation
4. **Evaluation:** uses the full graph for IC simulation

```python
# Correct usage
G_train, G_val, G_test = dataset.temporal_split()
data = dataset.prepare_pytorch_data(G_train)  # Use G_train!
```

### Statistical Significance
Results reported as mean ± std over 5 random initializations with fixed temporal split.

| Dataset | t-statistic | p-value |
|---------|-------------|---------|
| Cora-ML | 4.21 | 0.007 |
| Weibo | 5.67 | 0.002 |
| DBLP | 7.12 | 0.001 |

---

## 📖 Citation
```bibtex
@inproceedings{graphragim2026,
  title     = {GraphRAG-IM: Retrieval-Augmented Graph Neural Networks for Influence Maximization},
  author    = {Anonymous},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026}
}
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Sentence-Transformers](https://www.sbert.net/)
- [G-Retriever](https://github.com/XiaoxinHe/G-Retriever) (PCST retrieval)
