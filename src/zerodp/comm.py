"""Low-level distributed helpers shared by the DDP and FSDP implementations.

Everything here is a thin wrapper around `torch.distributed`. The point
of factoring it out is so `ddp.py` and `fsdp.py` never call `torch.distributed`
directly -- they go through this module, which also degrades gracefully
to single-process (world_size == 1) operation so the same model code
runs unmodified on a laptop and on a multi-GPU node.
"""
from __future__ import annotations

import os
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def init_distributed(backend: str = "nccl") -> torch.device:
    """Initialise the default process group from torchrun-supplied env vars.

    Safe to call even when not launched under torchrun: in that case
    `WORLD_SIZE` defaults to 1, no process group is created, and a plain
    `cuda`/`cpu` device is returned.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    local_rank = get_local_rank()
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def barrier() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()


def chain_params_and_buffers(module: nn.Module) -> Iterator[torch.Tensor]:
    for p in module.parameters():
        yield p
    for b in module.buffers():
        yield b


def broadcast_module(module: nn.Module, src: int = 0) -> None:
    """Broadcast every parameter and buffer from `src` to all ranks, in place.

    Called once at wrapper construction time so that every rank starts
    from bit-identical weights, regardless of how each rank's local
    `nn.Module.__init__` happened to seed its own RNG.
    """
    if not is_dist_available_and_initialized():
        return
    for tensor in chain_params_and_buffers(module):
        dist.broadcast(tensor.data, src=src)
