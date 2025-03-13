from __future__ import annotations

from functools import partial, cache
from collections import namedtuple
from typing import Callable, List

import torch
from torch import Tensor, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.amp import autocast

def exists(val) -> bool:
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    # Use a float value for p.
    return F.normalize(t, p=2.0, dim=dim, eps=eps)

def safe_div(num: Tensor, den: Tensor, eps: float = 1e-6) -> Tensor:
    return num / den.clamp(min=eps)

def pack_one(t: Tensor, pattern: str) -> list:
    """
    Emulates the functionality of einops' pack for the pattern '* d'.
    Flattens all dimensions of tensor `t` except for the last one.

    Args:
        t (Tensor): The input tensor.
        pattern (str): A string pattern (ignored in this implementation).

    Returns:
        list: A list containing two elements:
            [0]: The reshaped tensor of shape (-1, d).
            [1]: The original shape of the tensor as a list of ints.
    """
    original_shape: List[int] = list(t.shape)
    packed = t.reshape(-1, t.shape[-1])
    return [packed, original_shape]

def unpack_one(to_unpack: Tensor, original_shape: List[int]) -> Tensor:
    """
    Restores a tensor to its original shape.

    Args:
        to_unpack (Tensor): The tensor to reshape.
        original_shape (List[int]): The original shape to restore.

    Returns:
        Tensor: The reshaped tensor.
    """
    # Convert the list back to a tuple for reshape.
    return to_unpack.reshape(tuple(original_shape))

def efficient_rotation_trick_transform(u: Tensor, q: Tensor, e: Tensor) -> Tensor:
    """
    Implements the transformation described in section 4.2 of:
    https://arxiv.org/abs/2410.06424
    """
    e = e[:, None, :]
    w = l2norm(u + q, dim=1).detach()
    return (
        e -
        2 * (e @ w[:, :, None] @ w[:, None, :]) +
        2 * (e @ u[:, :, None].detach() @ q[:, None, :].detach())
    )

def rotate_to(src: Tensor, tgt: Tensor) -> Tensor:
    """
    Applies the rotation trick (STE) from https://arxiv.org/abs/2410.06424
    to allow gradients to flow through the VQ layer.

    Args:
        src (Tensor): Source tensor.
        tgt (Tensor): Target tensor.

    Returns:
        Tensor: The rotated tensor.
    """
    pack_result_src = pack_one(src, '* d')
    src_packed = pack_result_src[0]
    src_shape = pack_result_src[1]

    pack_result_tgt = pack_one(tgt, '* d')
    tgt_packed = pack_result_tgt[0]
    # The original shape of tgt is not used in this function.

    norm_src = src_packed.norm(2.0, -1, True)
    norm_tgt = tgt_packed.norm(2.0, -1, True)

    rotated_tgt = efficient_rotation_trick_transform(
        safe_div(src_packed, norm_src),
        safe_div(tgt_packed, norm_tgt),
        src_packed
    ).squeeze()

    rotated = rotated_tgt * safe_div(norm_tgt, norm_src).detach()

    return unpack_one(rotated, src_shape)
