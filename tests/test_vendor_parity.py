"""Confirms the vendored model architecture (models/vendor/) exactly
reproduces what the real checkpoints expect: every parameter name and shape
in the checkpoint's raw state_dict is matched exactly by the vendored
class's own state_dict. Doesn't need a GPU or network access -- catches
architecture drift (a missing layer, a renamed parameter) independently of
tests/test_smoke.py's full forward pass."""
import io
import json
import zipfile

import pytest
import torch

from coastdiff.config import load_config
from coastdiff.models.checkpoint_loader import find_latest_checkpoint
from coastdiff.models.vendor.edm_precond import EDMPrecondSuperResolution
from coastdiff.models.vendor.regression_unet import CorrDiffRegressionUNet


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _assert_state_dict_matches_checkpoint(cls, ckpt_path):
    with zipfile.ZipFile(ckpt_path) as z:
        args = json.loads(z.read("args.json").decode())["__args__"]
        real_sd = torch.load(io.BytesIO(z.read("model.pt")), map_location="cpu")

    model = cls(**args)
    my_sd = model.state_dict()

    real_keys, my_keys = set(real_sd), set(my_sd)
    assert real_keys == my_keys, (
        f"missing={real_keys - my_keys} extra={my_keys - real_keys}"
    )
    mismatches = [k for k in real_keys if real_sd[k].shape != my_sd[k].shape]
    assert not mismatches, f"shape mismatches: {mismatches}"


def test_regression_unet_matches_checkpoint(cfg):
    ckpt = find_latest_checkpoint(cfg["regression_checkpoint_dir"], "CorrDiffRegressionUNet")
    _assert_state_dict_matches_checkpoint(CorrDiffRegressionUNet, ckpt)


def test_edm_precond_matches_checkpoint(cfg):
    ckpt = find_latest_checkpoint(cfg["diffusion_checkpoint_dir"], "EDMPrecondSuperResolution")
    _assert_state_dict_matches_checkpoint(EDMPrecondSuperResolution, ckpt)
