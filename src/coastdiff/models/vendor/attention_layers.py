# SPDX-FileCopyrightText: Adapted from NVIDIA PhysicsNeMo (Apache-2.0) -- see NOTICE.
"""`UNetAttention`, extracted from physicsnemo.nn.module.attention_layers."""
import math
from typing import Any, Dict

import torch

from .conv_layers import Conv2d
from .group_norm import get_group_norm

class UNetAttention(torch.nn.Module):
    """
    Self-attention block used in U-Net-style architectures, such as DDPM++, NCSN++, and ADM.
    Applies GroupNorm followed by multi-head self-attention and a projection layer.

    Parameters
    ----------
    out_channels : int
        Number of channels :math:`C` in the input and output feature maps.
    num_heads : int
        Number of attention heads. Must be a positive integer.
    eps : float, optional, default=1e-5
        Epsilon value for numerical stability in GroupNorm.
    init_zero : dict, optional, default={'init_weight': 0}
        Initialization parameters with zero weights for certain layers.
    init_attn : dict, optional, default=None
        Initialization parameters specific to attention mechanism layers.
        Defaults to 'init' if not provided.
    init : dict, optional, default={}
        Initialization parameters for convolutional and linear layers.
    use_apex_gn : bool, optional, default=False
        A boolean flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Need to set this as False on cpu.
    amp_mode : bool, optional, default=False
        A boolean flag indicating whether mixed-precision (AMP) training is enabled.
    fused_conv_bias: bool, optional, default=False
        A boolean flag indicating whether bias will be passed as a parameter of conv2d.


    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, H, W)`, where :math:`B` is batch
        size, :math:`C` is `out_channels`, and :math:`H, W` are spatial
        dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of the same shape as input: :math:`(B, C, H, W)`.
    """

    def __init__(
        self,
        *,
        out_channels: int,
        num_heads: int,
        eps: float = 1e-5,
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
        init: Dict[str, Any] = dict(),
        use_apex_gn: bool = False,
        amp_mode: bool = False,
        fused_conv_bias: bool = False,
    ) -> None:
        super().__init__()
        # Parameters validation
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"`num_heads` must be a positive integer, but got {num_heads}"
            )
        if out_channels % num_heads != 0:
            raise ValueError(
                f"`out_channels` must be divisible by `num_heads`, but got {out_channels} and {num_heads}"
            )
        self.num_heads = num_heads
        self.norm = get_group_norm(
            num_channels=out_channels,
            eps=eps,
            use_apex_gn=use_apex_gn,
            amp_mode=amp_mode,
        )
        self.qkv = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * 3,
            kernel=1,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **(init_attn if init_attn is not None else init),
        )
        self.proj = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=1,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **init_zero,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1: torch.Tensor = self.qkv(self.norm(x))

        # # NOTE: V1.0.1 implementation
        # q, k, v = x1.reshape(
        #     x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
        # ).unbind(2)
        # w = AttentionOp.apply(q, k)
        # attn = torch.einsum("nqk,nck->ncq", w, v)

        qkv = (
            x1.reshape(x.shape[0], self.num_heads, x.shape[1] // self.num_heads, 3, -1)
        ).permute(0, 1, 4, 3, 2)
        (q, k, v) = (qkv[..., i, :] for i in range(3))
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(k.shape[-1])
        )
        attn = attn.transpose(-1, -2)

        x: torch.Tensor = self.proj(attn.reshape(*x.shape)).add_(x)
        return x
