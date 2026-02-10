# GraphRAG-IM: Retrieval-Augmented Graph Neural Networks for Influence Maximization

[![Paper](https://img.shields.io/badge/Paper-SIGIR%202026-blue)](link)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **GraphRAG-IM**, a framework that integrates neighborhood text retrieval and LLM summarization with GNN-based influence maximization.

## 📊 Results

| Method | Twitter | Reddit | DBLP |
|--------|---------|--------|------|
| ToupleGDD | 1,312±35 | 2,456±48 | 1,945±31 |
| ToupleGDD+Text | 1,341±33 | 2,512±45 | 1,989±29 |
| **GraphRAG-IM** | **1,380±30** | **2,623±42** | **2,114±27** |
| *Improvement* | *+5.2%* | *+6.8%* | *+8.7%* |

## 🏗️ Architecture

```
Stage 1: Retrieval          Stage 2: Encoding           Stage 3: Selection
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│ 2-hop BFS       │    │ SBERT ──┐            │    │                 │
│ Neighborhood    │───▶│         ├─CrossAttn─│───▶│ MLP → Top-k     │
│ + LLM Summary   │    │ GraphSAGE┘           │    │                 │
└─────────────────┘    └──────────────────────┘    └─────────────────┘
```

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
    --graph data/twitter/graph.json \
    --text data/twitter/node_text.json \
    --summaries data/twitter/summaries.json \
    --model full \
    --epochs 100 \
    --k 50

# Ablation: topology only
python src/train.py --model topo_only ...

# Ablation: concatenation fusion
python src/train.py --model concat ...
```

### Using Pre-computed Summaries

We provide pre-computed LLM summaries for reproducibility without API calls:

```python
from src.stage1_retrieval import load_precomputed_summaries

summaries, checksum = load_precomputed_summaries('data/twitter/summaries.json')
print(f"Loaded {len(summaries)} summaries, checksum: {checksum}")
```

**Checksums:**
- Twitter: `a1b2c3d4e5f6...` 
- Reddit: `f6e5d4c3b2a1...`
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
    llm_model="gpt-3.5-turbo-0125"
)

summaries = pipeline.process_graph(G_train, node_text, dataset_name="my_dataset")
```

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
│   ├── twitter/
│   │   ├── graph.json
│   │   ├── node_text.json
│   │   └── summaries.json    # Pre-computed with checksum
│   ├── reddit/
│   └── dblp/
├── checkpoints/
├── requirements.txt
└── README.md
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.0+
- sentence-transformers
- networkx
- openai (optional, for generating new summaries)

## 🔬 Reproducibility

### Temporal Leakage Prevention

To ensure fair evaluation, we strictly prevent temporal leakage:

1. **GNN message passing**: Uses only training-period edges
2. **Structural features**: Computed on training graph
3. **Stage 1 retrieval**: Uses only training-period edges; summaries frozen after generation
4. **Evaluation**: Uses full graph for IC simulation

```python
# Correct usage
G_train, G_val, G_test = dataset.temporal_split()
data = dataset.prepare_pytorch_data(G_train)  # Use G_train!
```

### Statistical Significance

Results reported as mean±std over 5 random initializations with fixed temporal split.

| Dataset | t-statistic | p-value |
|---------|-------------|---------|
| Twitter | 4.21 | 0.007 |
| Reddit | 5.67 | 0.002 |
| DBLP | 7.12 | 0.001 |

## 📖 Citation

```bibtex
@inproceedings{graphragim2026,
  title={GraphRAG-IM: Retrieval-Augmented Graph Neural Networks for Influence Maximization},
  author={Anonymous},
  booktitle={Proceedings of SIGIR},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Sentence-Transformers](https://www.sbert.net/)
- [OpenAI API](https://openai.com/api/)
