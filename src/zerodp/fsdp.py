"""A simplified, from-scratch reimplementation of FSDP-style ZeRO-3 sharding.

Each `FSDPUnit` wraps one `nn.Module` and shards every one of its
parameters evenly across the process group: every rank holds only
`1 / world_size` of the unit's parameter data at rest. On every forward
call the local shards are all-gathered back into full parameters
(through a custom `autograd.Function` so that backward automatically
reduce-scatters the resulting gradient into per-rank shards), the
wrapped module runs as normal against the full parameters, and the
python-level references to the full tensors are then dropped.

Gradients are averaged across ranks (reduce-scatter-sum followed by a
divide), matching this repo's DDP implementation and upstream FSDP's
default -- not a raw sum.

Known simplifications versus upstream `torch.distributed.fsdp.FSDP`:
  * No recursive auto-wrap policy. The caller decides which submodules
    become their own `FSDPUnit` (see `wrap_module_list` and
    `auto_wrap_children` below for two common patterns).
  * No backward-time "discard then re-gather exactly when needed"
    scheduling. Dropping the python reference after forward lets
    autograd free each gathered tensor as soon as its own backward
    finishes, but the full tensors for *every* unit that participated
    in forward remain reachable (via the autograd graph) until that
    unit's own backward executes. Peak memory is therefore higher than
    upstream FSDP's backward pre-fetch design, though still far below
    replicating every parameter on every rank.
  * No CPU offload, no dedicated mixed-precision parameter storage, no
    `summon_full_params` debugging context manager.
  * Any parameter that lives outside of some `FSDPUnit` is never
    synchronised by this module at all -- it will silently diverge
    across ranks. `auto_wrap_children` exists specifically to avoid
    leaving parameters unwrapped by accident.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from .comm import broadcast_module, get_rank, get_world_size, is_dist_available_and_initialized


class _AllGatherFlat(torch.autograd.Function):
    """Forward: all-gather a local shard into the full flat parameter.

    Backward: reduce-scatter (sum) the incoming gradient back into the
    shard that belongs to this rank, then divide by world size so the
    result is an average, not a sum, across ranks.
    """

    @staticmethod
    def forward(ctx, local_shard: torch.Tensor, world_size: int, group: Any) -> torch.Tensor:
        ctx.world_size = world_size
        ctx.group = group
        ctx.shard_numel = local_shard.numel()
        if world_size == 1:
            return local_shard.clone()
        out = torch.empty(
            world_size * local_shard.numel(), dtype=local_shard.dtype, device=local_shard.device
        )
        dist.all_gather_into_tensor(out, local_shard.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        world_size = ctx.world_size
        if world_size == 1:
            return grad_output, None, None
        grad_input = torch.empty(
            ctx.shard_numel, dtype=grad_output.dtype, device=grad_output.device
        )
        dist.reduce_scatter_tensor(
            grad_input, grad_output.contiguous(), op=dist.ReduceOp.SUM, group=ctx.group
        )
        grad_input.div_(world_size)
        return grad_input, None, None


def _pad_to_multiple(flat: torch.Tensor, multiple: int) -> torch.Tensor:
    remainder = flat.numel() % multiple
    if remainder == 0:
        return flat
    pad = multiple - remainder
    return torch.cat([flat, flat.new_zeros(pad)])


class FSDPUnit(nn.Module):
    """Wraps `module`, flattening and evenly sharding all of its parameters."""

    def __init__(self, module: nn.Module, group: Optional[Any] = None):
        super().__init__()
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.group = group

        if is_dist_available_and_initialized() and self.world_size > 1:
            broadcast_module(module, src=0)

        params = [p for p in module.parameters() if p.requires_grad]
        self._param_names = [name for name, p in module.named_parameters() if p.requires_grad]
        self._shapes = [p.shape for p in params]
        self._numels = [p.numel() for p in params]

        flat = (
            torch.cat([p.detach().reshape(-1) for p in params])
            if params
            else torch.empty(0)
        )
        flat = _pad_to_multiple(flat, max(self.world_size, 1))
        self._padded_numel = flat.numel()
        shard_numel = self._padded_numel // max(self.world_size, 1)
        shard = flat[self.rank * shard_numel : (self.rank + 1) * shard_numel].clone()

        self.local_shard = nn.Parameter(shard)
        self.module = module

        # Replace each real nn.Parameter with a plain (initially absent)
        # attribute; _gather_and_bind fills it in with an autograd-tracked
        # view into the all-gathered flat buffer on every forward call.
        for name in self._param_names:
            self._delete_param(self.module, name)

    @staticmethod
    def _delete_param(module: nn.Module, name: str) -> None:
        *parents, leaf = name.split(".")
        target = module
        for p in parents:
            target = getattr(target, p)
        delattr(target, leaf)

    @staticmethod
    def _set_attr(module: nn.Module, name: str, value: Any) -> None:
        *parents, leaf = name.split(".")
        target = module
        for p in parents:
            target = getattr(target, p)
        setattr(target, leaf, value)

    def _gather_and_bind(self) -> None:
        full = _AllGatherFlat.apply(self.local_shard, self.world_size, self.group)
        full = full[: sum(self._numels)]
        offset = 0
        for name, shape, numel in zip(self._param_names, self._shapes, self._numels):
            view = full[offset : offset + numel].view(shape)
            self._set_attr(self.module, name, view)
            offset += numel

    def _unbind(self) -> None:
        for name in self._param_names:
            self._set_attr(self.module, name, None)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._gather_and_bind()
        out = self.module(*args, **kwargs)
        self._unbind()
        return out


def wrap_module_list(module_list: nn.ModuleList, group: Optional[Any] = None) -> nn.ModuleList:
    """Replace every entry in `module_list` with an `FSDPUnit` wrapping it.

    The common case: `model.blocks = wrap_module_list(model.blocks)` for a
    `nn.ModuleList` of transformer blocks, giving one FSDP unit per block.
    """
    return nn.ModuleList([FSDPUnit(m, group=group) for m in module_list])


def auto_wrap_children(
    module: nn.Module, skip: tuple[str, ...] = (), group: Optional[Any] = None
) -> None:
    """In place: wrap every immediate child of `module` that owns at least
    one trainable parameter in its own `FSDPUnit`, except names in `skip`.

    Use this to make sure parameters living outside whatever you wrapped
    by hand (e.g. token/position embeddings and the output head sitting
    alongside a `ModuleList` of transformer blocks) still end up inside
    some FSDP unit. Anything left outside of one is never synchronised by
    this module and will silently diverge across ranks.
    """
    for name, child in list(module.named_children()):
        if name in skip:
            continue
        if any(p.requires_grad for p in child.parameters(recurse=True)):
            setattr(module, name, FSDPUnit(child, group=group))
