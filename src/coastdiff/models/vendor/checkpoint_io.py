"""Minimal `.mdlus` checkpoint loader.

A `.mdlus` file (physicsnemo's own checkpoint format) is a plain zip with:
  args.json     -- {"__name__": <class name>, "__module__": ..., "__args__": {...init kwargs...}}
  metadata.json -- {"physicsnemo_version": ..., "mdlus_file_version": ...}
  model.pt      -- a plain torch.save'd state_dict

`__module__` is ignored here (we always instantiate the caller-supplied
`cls` directly, not whatever class the checkpoint happens to name) --
`args.json` still tells us exactly which constructor kwargs to use.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: str | Path, cls: type, device: torch.device | str = "cpu",
                     override_args: dict[str, Any] | None = None) -> torch.nn.Module:
    with zipfile.ZipFile(path) as z:
        args = json.loads(z.read("args.json").decode())["__args__"]
        metadata = json.loads(z.read("metadata.json").decode())
        state_dict = torch.load(io.BytesIO(z.read("model.pt")), map_location=device)

    version = metadata.get("mdlus_file_version", "0.0.0")
    args = cls._backward_compat_arg_mapper(version, dict(args))
    if override_args:
        args.update(override_args)

    model = cls(**args)
    model.load_state_dict(state_dict)
    return model
