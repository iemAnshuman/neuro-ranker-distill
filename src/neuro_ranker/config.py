from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_SECTIONS = ("data", "teacher", "student", "evaluation")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    if config.get("profile") not in {"fixture", "smoke", "artifact"}:
        raise ValueError("profile must be fixture, smoke, or artifact")
    for section in REQUIRED_SECTIONS:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing config section: {section}")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    return config
