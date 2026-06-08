from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    text_dim: int = 384
    graph_hidden: int = 128
    graph_layers: int = 3
    attn_dim: int = 64
    attn_slots: int = 6
    dropout: float = 0.1
    mlp_hidden: int = 64
    mlp_dropout: float = 0.3
    sbert_model: str = "all-MiniLM-L6-v2"


@dataclass
class RetrievalConfig:
    n_hops: int = 2
    alpha: float = 0.5
    gamma: float = 1.0
    structural_prior: str = "pagerank"
    max_nodes: int = 16
    backend: str = "greedy"
    summarizer: str = "offline"


@dataclass
class DiffusionConfig:
    p0: float = 0.1
    rho_mode: str = "cosine"
    rho_temperature: float = 1.0
    n_sims: int = 1000
    directed: bool = False


@dataclass
class SelectionConfig:
    k: int = 50
    method: str = "dpp"
    theta: float = 0.1


@dataclass
class TrainingConfig:
    scoring: str = "heuristic"          # 'heuristic' (offline) | 'neural' (torch)
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    label_top_pct: float = 0.10
    label_membership: float = 0.50
    label_n_configs: int = 100
    label_sims: int = 1000


@dataclass
class DataConfig:
    name: str = "synthetic"
    root: str = "data/processed"
    n_communities: int = 6
    nodes_per_community: int = 60
    p_in: float = 0.10
    p_out: float = 0.004
    embedding_dim: int = 64
    topic_separation: float = 2.0


@dataclass
class LoggingConfig:
    log_interval: int = 25
    output_dir: str = "outputs"
    eval_sims: int = 300


@dataclass
class GraphRAGIMConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        return cls(
            seed=config_dict.get("seed", 42),
            model=ModelConfig(**config_dict.get("model", {})),
            retrieval=RetrievalConfig(**config_dict.get("retrieval", {})),
            diffusion=DiffusionConfig(**config_dict.get("diffusion", {})),
            selection=SelectionConfig(**config_dict.get("selection", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "model": self.model.__dict__,
            "retrieval": self.retrieval.__dict__,
            "diffusion": self.diffusion.__dict__,
            "selection": self.selection.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "logging": self.logging.__dict__,
        }

    def to_yaml(self, yaml_path: str):
        with open(yaml_path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
