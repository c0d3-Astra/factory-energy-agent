"""Configuration: assumptions from config.yaml, runtime switches from the environment."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict:
    path = path or os.getenv("CONFIG_PATH") or str(ROOT / "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if os.getenv("EMISSION_FACTOR_KG_PER_KWH"):
        cfg["emission_factor_kg_per_kwh"] = float(os.environ["EMISSION_FACTOR_KG_PER_KWH"])
    if os.getenv("FACTORY_TIMEZONE"):
        cfg["factory_timezone"] = os.environ["FACTORY_TIMEZONE"]
    return cfg


def with_overrides(cfg: dict, **baseline_overrides) -> dict:
    """Copy of cfg with baseline settings changed (used by the evaluation comparison)."""
    out = copy.deepcopy(cfg)
    out["baseline"].update(baseline_overrides)
    return out


def db_path() -> str:
    return os.getenv("DB_PATH", str(ROOT / "data" / "energy.duckdb"))


def llm_settings() -> dict:
    return {
        "mock": os.getenv("MOCK_LLM", "true").lower() in ("1", "true", "yes"),
        "provider": os.getenv("LLM_PROVIDER", "ollama").lower(),
        "model": os.getenv("LLM_MODEL", ""),
    }
