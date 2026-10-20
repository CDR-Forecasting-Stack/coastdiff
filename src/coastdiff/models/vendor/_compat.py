# SPDX-FileCopyrightText: Adapted from NVIDIA PhysicsNeMo (Apache-2.0) -- see NOTICE.
"""Minimal compatibility shims replacing the small slice of NVIDIA
PhysicsNeMo's `physicsnemo.core` functionality the vendored architecture
code below actually calls -- so this package no longer requires a
PhysicsNeMo installation at all.

Verified sufficient two ways: (1) every parameter name/shape produced by
the vendored classes matches the real checkpoint's raw state_dict exactly
(see tests/test_vendor_parity.py), and (2) a full forward pass through the
vendored regression+diffusion models, loaded from the real checkpoints,
matches the original PhysicsNeMo-backed implementation bit-for-bit on the
same input.
"""
import torch


class Module(torch.nn.Module):
    """Stand-in for physicsnemo.core.module.Module. The vendored classes
    call: `super().__init__(meta=...)` (the `meta` object itself is never
    read anywhere -- it's PhysicsNeMo's own model-registry metadata, unused
    here); `super()._backward_compat_arg_mapper(version, args)` (a no-op
    passthrough at the base -- version-specific overrides, where present,
    are defined directly on the subclasses and preserved verbatim); and
    `net.device` (via `generate.regression_step`), which the real
    PhysicsNeMo Module derives from a registered non-persistent buffer --
    reproduced here for the same reason. Checkpoint I/O itself is handled
    by checkpoint_io.py, not by this class."""

    _overridable_args: set = set()

    def __init__(self, meta=None):
        super().__init__()
        self.register_buffer("device_buffer", torch.empty(0), persistent=False)

    @classmethod
    def _backward_compat_arg_mapper(cls, version, args):
        return args

    @property
    def device(self) -> torch.device:
        return self.device_buffer.device


class ModelMetaData:
    """Placeholder for physicsnemo.core.meta.ModelMetaData. Never read at
    runtime (see Module above) -- subclasses may declare whatever fields
    the original did, but nothing here depends on their values."""


def _validate_amp(amp_mode: bool) -> None:
    """Raise if `amp_mode` is False but PyTorch autocast (CPU or CUDA) is
    active."""
    try:
        cuda_amp = bool(torch.is_autocast_enabled())
    except AttributeError:  # very old PyTorch
        cuda_amp = False
    try:
        cpu_amp = bool(torch.is_autocast_enabled("cpu"))
    except AttributeError:
        cpu_amp = False

    if not amp_mode and (cuda_amp or cpu_amp):
        active = []
        if cuda_amp:
            active.append("cuda")
        if cpu_amp:
            active.append("cpu")
        raise RuntimeError(
            f"amp_mode=False but torch autocast is enabled on: {', '.join(active)}. "
            "Disable autocast for this region or set amp_mode=True if mixed precision is intended."
        )


class _NoOpContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _NvtxShim:
    """No-op stand-in for the `nvtx` profiling-annotation library (purely
    cosmetic profiler markers in the original code; no effect on
    computation)."""

    @staticmethod
    def annotate(*args, **kwargs):
        return _NoOpContext()


nvtx = _NvtxShim()


class LegacyFeatureWarning(DeprecationWarning):
    pass


class ExperimentalFeatureWarning(UserWarning):
    pass


class GridPatching2D:
    """Placeholder for physicsnemo.diffusion.multi_diffusion.GridPatching2D.
    Multi-patch diffusion is never used by this package (always
    `patching=None`), so this type is never actually instantiated -- it
    exists only to satisfy isinstance/type-hint references in the vendored
    sampler code."""


class _EDMPrecondSentinel:
    """Placeholder for physicsnemo's `EDMPrecond` class -- a different
    model variant than `EDMPrecondSuperResolution`, which is what this
    package always uses. The vendored sampler's `isinstance(net,
    EDMPrecond)` check is always False for our model in the original
    PhysicsNeMo implementation too; this placeholder preserves that exact
    behavior (the sampler always takes the same branch it always took).

    `EDMPrecondSuperResolution.round_sigma` delegates to
    `EDMPrecond.round_sigma` (called at runtime by the sampler), so that
    one static method is reproduced exactly (verbatim one-liner from the
    original `EDMPrecond` class) rather than left as a dead placeholder."""

    @staticmethod
    def round_sigma(sigma):
        return torch.as_tensor(sigma)


EDMPrecond = _EDMPrecondSentinel
