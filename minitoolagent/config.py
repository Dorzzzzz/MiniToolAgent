from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "llm.yaml"


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key: str = ""
    base_url: str = "https://api.siliconflow.cn/v1"
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: int = 120
    brave_api_key: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> Config:
        path = Path(path) if path else _DEFAULT_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
