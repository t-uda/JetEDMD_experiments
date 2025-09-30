from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml
import os, json


@dataclass
class ExperimentConfig:
    system: str
    params: Dict[str, Any]
    dt_true: float
    Tfast: float
    r_list: List[float]
    noise: Dict[str, Any] = field(default_factory=dict)
    sampling: Dict[str, Any] = field(default_factory=dict)
    splits: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.6, "val": 0.2, "test": 0.2}
    )
    seeds: List[int] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    # Optional fields for specific experiments
    extras: Dict[str, Any] = field(default_factory=dict)


def load_yaml(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Normalize fields
    data.setdefault("noise", {})
    data.setdefault("sampling", {})
    data.setdefault("splits", {"train": 0.6, "val": 0.2, "test": 0.2})
    data.setdefault("metrics", [])
    data.setdefault("extras", {})
    return ExperimentConfig(**data)


def dump_config(cfg: ExperimentConfig, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.__dict__, f, allow_unicode=True, sort_keys=False)
