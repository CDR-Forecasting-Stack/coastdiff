"""Config loading -- configs/default.yaml is the single source of truth for
checkpoint directories, stats path, and defaults."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PKG_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _PKG_ROOT / "configs" / "default.yaml"

# Asset keys: a relative path here is resolved against the installed package
# directory (so it works regardless of the caller's cwd), not the caller's cwd;
# "~" is expanded, so checkpoints can live in a per-user cache. `output_dir`
# is deliberately NOT in this list: it should resolve relative to cwd, like
# any normal CLI output path.
_ASSET_KEYS = ("regression_checkpoint_dir", "diffusion_checkpoint_dir", "stats_path")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in _ASSET_KEYS:
        val = cfg.get(key)
        if val:
            p = Path(val).expanduser()
            cfg[key] = str(p if p.is_absolute() else _PKG_ROOT / p)
    return cfg
