"""Download the pretrained checkpoints from Zenodo on first use.

The two ~319 MB checkpoints are published at https://doi.org/10.5281/zenodo.23510727
and are not stored in this git repository. `ensure_checkpoint` leaves any checkpoint already
present in the target directory alone (so a locally trained or manually placed one wins) and
downloads the published one, verified by SHA-256, only when the directory has none.
"""
from __future__ import annotations

import hashlib
import ssl
import urllib.request
from pathlib import Path

import certifi

ZENODO_DOI = "10.5281/zenodo.23510727"
_BASE_URL = "https://zenodo.org/records/23510727/files"

# prefix -> (file name, size in bytes, sha256)
CHECKPOINTS = {
    "CorrDiffRegressionUNet": (
        "CorrDiffRegressionUNet.0.5000000.mdlus",
        319085619,
        "f666b2b5b42d1601e7372d7b59194a6411519271052b07ce39ad7685a644a890",
    ),
    "EDMPrecondSuperResolution": (
        "EDMPrecondSuperResolution.0.50000000.mdlus",
        319122577,
        "538b8f3ac7f81e380f11aea4bd79d2dee6f55aeb8dbf88ca154d1ad13b093b9d",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_checkpoint(ckpt_dir: str | Path, prefix: str) -> None:
    """Make sure `ckpt_dir` holds a `<prefix>.*.mdlus` checkpoint, downloading the published one if not."""
    ckpt_dir = Path(ckpt_dir)
    if any(ckpt_dir.rglob(f"{prefix}.*.mdlus")):
        return
    name, size, sha256 = CHECKPOINTS[prefix]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dest = ckpt_dir / name
    part = dest.with_suffix(dest.suffix + ".part")
    url = f"{_BASE_URL}/{name}?download=1"
    print(f"Downloading {name} ({size / 1e6:.0f} MB) from Zenodo, doi:{ZENODO_DOI} -> {dest}", flush=True)
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx) as resp, open(part, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    if part.stat().st_size != size or _sha256(part) != sha256:
        part.unlink()
        raise RuntimeError(
            f"Downloaded {name} failed its size/SHA-256 check; try again, or download it by hand from "
            f"https://doi.org/{ZENODO_DOI} into {ckpt_dir}."
        )
    part.rename(dest)
