# SPDX-FileCopyrightText: Adapted from NVIDIA PhysicsNeMo (Apache-2.0) -- see NOTICE.
"""`Linear` layer, extracted from physicsnemo.nn.module.fully_connected_layers."""
from typing import Literal

import torch
from torch import Tensor

from ._compat import Module, _validate_amp
from .weight_init import _weight_init

class Linear(Module):
    r"""Fully connected (dense) layer with customizable initialization.

    The layer's weights and biases can be initialized using custom strategies
    like ``"kaiming_normal"``, and scaled by ``init_weight`` and ``init_bias``.
    Parameters
    ----------
    in_features : int
        Size of each input sample :math:`D_{in}`.
    out_features : int
        Size of each output sample :math:`D_{out}`.
    bias : bool, optional, default=True
        If ``True``, adds a learnable bias to the output. If ``False``, the layer
        will not learn an additive bias.
    init_mode : str, optional, default="kaiming_normal"
        The initialization mode for weights and biases. Supported modes are:
        ``"xavier_uniform"``, ``"xavier_normal"``, ``"kaiming_uniform"``,
        ``"kaiming_normal"``.
    init_weight : float, optional, default=1
        A scaling factor to multiply with the initialized weights.
    init_bias : float, optional, default=0
        A scaling factor to multiply with the initialized biases.
    amp_mode : bool, optional, default=False
        Whether mixed-precision (AMP) training is enabled.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(*, D_{in})` where :math:`*` denotes any
        number of leading batch dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(*, D_{out})`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mode: Literal[
            "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"
        ] = "kaiming_normal",
        init_weight: int = 1,
        init_bias: int = 0,
        amp_mode: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.amp_mode = amp_mode
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            _weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(_weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the linear layer."""
        weight, bias = self.weight, self.bias
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if self.weight is not None and self.weight.dtype != x.dtype:
                weight = self.weight.to(x.dtype)
            if self.bias is not None and self.bias.dtype != x.dtype:
                bias = self.bias.to(x.dtype)
        x = x @ weight.t()
        if self.bias is not None:
            x = x.add_(bias)
        return x
