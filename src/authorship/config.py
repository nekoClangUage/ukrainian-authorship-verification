import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class ModelConfig:
    embed_dim: int = 256
    hidden_dim: int = 384
    classifier_h1: int = 512
    classifier_h2: int = 256
    dropout: float = 0.3
    embed_dropout: float = 0.2


@dataclass
class TrainingConfig:
    lr: float = 0.001
    epochs: int = 5
    batch_size: int = 16
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    scheduler_min_lr: float = 1e-6
    grad_clip_max_norm: float = 1.0
    weight_decay: float = 1e-4


@dataclass
class DataConfig:
    max_seq_len: int = 1000
    min_freq: int = 2
    train_samples_per_author: int = 300
    val_samples_per_author: int = 150
    vocab_force_rebuild: bool = False
    divider_target_count: int = 700


@dataclass
class InferenceConfig:
    threshold: float = 0.65


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(
        default_factory=InferenceConfig
    )
    seed: int = 2026
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """
        Load config from a YAML file with grouped sections, falling back to defaults.
        If the file does not exist, returns a Config with default values.
        """
        cfg = cls()
        try:
            with Path(path).open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for section_name, sub_cfg in [
                ("model", cfg.model),
                ("training", cfg.training),
                ("data", cfg.data),
                ("inference", cfg.inference),
            ]:
                if section_name in data and isinstance(
                    data[section_name], dict
                ):
                    for key, value in data[
                        section_name
                    ].items():
                        if hasattr(sub_cfg, key):
                            setattr(sub_cfg, key, value)
            if "seed" in data:
                cfg.seed = data["seed"]
        except FileNotFoundError:
            pass
        return cfg


def get_config_path(default: Path) -> Path:
    """Returns --config <path> CLI argument or the default path."""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    return default


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
