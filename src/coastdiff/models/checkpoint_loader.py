"""Checkpoint loading and model execution (vendored two-stage CorrDiff --
regression UNet + EDM diffusion residual, originally from NVIDIA
PhysicsNeMo, Apache-2.0, see NOTICE). No PhysicsNeMo installation is
required; see models/vendor/ for the vendored architecture/sampler code and
docs/ARCHITECTURE.md for how its correctness is verified."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch

from ..data.channels import OUT_CH
from .download import ensure_checkpoint
from .vendor import checkpoint_io
from .vendor.edm_precond import EDMPrecondSuperResolution
from .vendor.generate import diffusion_step, regression_step
from .vendor.regression_unet import CorrDiffRegressionUNet
from .vendor.sampler import stochastic_sampler


def find_latest_checkpoint(ckpt_dir: str, prefix: str) -> str:
    """Recursive glob (checkpoints sometimes land one directory deeper)
    picking the highest training-iteration checkpoint -- so dropping a
    newer checkpoint into `ckpt_dir` is picked up automatically on the next
    run, no config change needed."""
    paths = list(Path(ckpt_dir).rglob(f"{prefix}.*.mdlus"))
    if not paths:
        raise FileNotFoundError(f"No {prefix}*.mdlus checkpoints found under {ckpt_dir}")

    def iter_num(p: Path) -> int:
        try:
            return int(p.stem.split(".")[-1])
        except ValueError:
            return -1

    return str(max(paths, key=iter_num))


def init_single_process_distributed():
    """No-op: kept only so callers written against the old PhysicsNeMo-
    backed loader (which needed to force PhysicsNeMo's DistributedManager
    into single-process mode to avoid a multi-rank rendezvous hang) don't
    need to change. The vendored code here has no distributed-training
    machinery to initialize in the first place."""


class ModelPair:
    """Loaded regression (+ optional diffusion) networks, ready for inference."""

    def __init__(self, net_reg, net_res, sampler_fn, device: torch.device,
                 reg_ckpt: str, res_ckpt: str | None):
        self.net_reg = net_reg
        self.net_res = net_res
        self.sampler_fn = sampler_fn
        self.device = device
        self.reg_ckpt = reg_ckpt
        self.res_ckpt = res_ckpt


def load_models(regression_ckpt_dir: str, diffusion_ckpt_dir: str, device: torch.device,
                 mode: str = "all", sampler: str = "stochastic", sampler_steps: int | None = None,
                 solver: str = "euler") -> ModelPair:
    if sampler != "stochastic":
        raise NotImplementedError(
            "Only the stochastic sampler is vendored (coastdiff's default and only "
            "sampler used in practice); pass sampler='stochastic'."
        )

    ensure_checkpoint(regression_ckpt_dir, "CorrDiffRegressionUNet")
    reg_ckpt = find_latest_checkpoint(regression_ckpt_dir, "CorrDiffRegressionUNet")
    net_reg = checkpoint_io.load_checkpoint(
        reg_ckpt, CorrDiffRegressionUNet, device=device, override_args={"use_apex_gn": False}
    )
    net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
    if hasattr(net_reg, "amp_mode"):
        net_reg.amp_mode = False

    net_res = None
    res_ckpt = None
    if mode == "all":
        ensure_checkpoint(diffusion_ckpt_dir, "EDMPrecondSuperResolution")
        res_ckpt = find_latest_checkpoint(diffusion_ckpt_dir, "EDMPrecondSuperResolution")
        net_res = checkpoint_io.load_checkpoint(
            res_ckpt, EDMPrecondSuperResolution, device=device, override_args={"use_apex_gn": False}
        )
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
        if hasattr(net_res, "amp_mode"):
            net_res.amp_mode = False

    sampler_fn = partial(stochastic_sampler, num_steps=sampler_steps or 18, patching=None)

    return ModelPair(net_reg, net_res, sampler_fn, device, reg_ckpt, res_ckpt)


HR_MEAN_CONDITIONING = True  # property of the trained checkpoint, not a runtime choice


def run_patch(models: ModelPair, x_norm: np.ndarray, num_ensembles: int,
              seed_batch_size: int, normalizer) -> tuple[np.ndarray, np.ndarray | None]:
    """x_norm: (C,256,256) normalized model input (coarse channels upsampled
    + invariants, already normalized). Returns (unet_phys (8,256,256),
    ens_phys (E,8,256,256) or None)."""
    img_lr = torch.from_numpy(x_norm[None]).to(device=models.device, dtype=torch.float32)
    img_lr = img_lr.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        image_reg = regression_step(
            net=models.net_reg, img_lr=img_lr, latents_shape=(1, len(OUT_CH), 256, 256),
        )
        unet_phys = normalizer.denormalize_output(image_reg[0].float().cpu().numpy())

        ens_phys = None
        if models.net_res is not None:
            mean_hr = image_reg[0:1] if HR_MEAN_CONDITIONING else None
            seeds = torch.arange(num_ensembles)
            rank_batches = list(seeds.split(seed_batch_size))
            image_res = diffusion_step(
                net=models.net_res, sampler_fn=models.sampler_fn,
                img_shape=(256, 256), img_out_channels=len(OUT_CH),
                rank_batches=rank_batches,
                img_lr=img_lr.expand(seed_batch_size, -1, -1, -1).to(memory_format=torch.channels_last),
                rank=0, device=models.device, mean_hr=mean_hr,
            )
            image_out = image_reg.expand(num_ensembles, -1, -1, -1) + image_res
            ens_phys = normalizer.denormalize_output(image_out.float().cpu().numpy())

    return unet_phys, ens_phys
