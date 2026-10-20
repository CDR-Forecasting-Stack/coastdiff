"""Loads the latest checkpoints and runs one forward pass on synthetic
input -- no network required, catches checkpoint/environment drift quickly.
Needs the bundled checkpoint files present, but no PhysicsNeMo install (see
models/vendor/) and no ERA5 store access."""
import numpy as np
import pytest
import torch

from coastdiff.config import load_config
from coastdiff.data.channels import IN_CH, INVARIANT_CH, OUT_CH
from coastdiff.data.grid import HR_SIZE, LR_SIZE, UPSAMPLE_FACTOR
from coastdiff.data.normalization import Normalizer
from coastdiff.data.transforms import zoom_extrapolate
from coastdiff.models.checkpoint_loader import (
    init_single_process_distributed, load_models, run_patch,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_zoom_extrapolate_shape():
    x = np.random.rand(3, LR_SIZE, LR_SIZE).astype(np.float32)
    y = zoom_extrapolate(x, UPSAMPLE_FACTOR)
    assert y.shape == (3, HR_SIZE, HR_SIZE)
    assert np.isfinite(y).all()


def test_checkpoint_forward_pass(cfg):
    init_single_process_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = Normalizer(cfg["stats_path"])
    models = load_models(cfg["regression_checkpoint_dir"], cfg["diffusion_checkpoint_dir"],
                          device, mode="all")

    rng = np.random.default_rng(0)
    lr = rng.normal(size=(len(IN_CH), LR_SIZE, LR_SIZE)).astype(np.float32)
    x = zoom_extrapolate(lr, UPSAMPLE_FACTOR)
    inv = rng.normal(size=(len(INVARIANT_CH), HR_SIZE, HR_SIZE)).astype(np.float32)
    x = np.concatenate([x, inv], axis=0)
    x_norm = normalizer.normalize_input(x).astype(np.float32)

    unet_phys, ens_phys = run_patch(models, x_norm, num_ensembles=2, seed_batch_size=1,
                                     normalizer=normalizer)
    assert unet_phys.shape == (len(OUT_CH), HR_SIZE, HR_SIZE)
    assert np.isfinite(unet_phys).all()
    assert ens_phys.shape == (2, len(OUT_CH), HR_SIZE, HR_SIZE)
    assert np.isfinite(ens_phys).all()
