"""Config loading -- configs/default.yaml is the single source of truth for
checkpoint directories, stats path, and defaults."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"

# Bundled-asset keys: a relative path here is resolved against the package
# root (so the package is portable -- works regardless of where it's
# unzipped or what the caller's cwd is), not the caller's cwd. `output_dir`
# is deliberately NOT in this list: it should resolve relative to cwd, like
# any normal CLI output path.
_ASSET_KEYS = ("regression_checkpoint_dir", "diffusion_checkpoint_dir", "stats_path")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in _ASSET_KEYS:
        val = cfg.get(key)
        if val and not Path(val).is_absolute():
            cfg[key] = str(_REPO_ROOT / val)
    return cfg
