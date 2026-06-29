"""Gradient bucketing for the from-scratch DDP implementation.

Mirrors the core idea behind `torch.nn.parallel.DistributedDataParallel`:
group parameter gradients into fixed-size buckets, flatten each bucket
into one contiguous buffer, and kick off a single async all-reduce per
bucket as soon as every gradient inside it has been computed. This
overlaps communication with the rest of backward instead of waiting
for the whole backward pass to finish before talking to the network.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Bucket:
    params: list[torch.nn.Parameter]
    buffer: torch.Tensor
    offsets: list[int]
    numels: list[int]
    pending: set[int] = field(default_factory=set)
    handle: object = None

    def mark_ready(self, index: int) -> bool:
        """Record that `params[index]`'s gradient is ready.

        Returns True once every gradient in the bucket has arrived.
        """
        self.pending.discard(index)
        return len(self.pending) == 0

    def reset(self) -> None:
        self.pending = set(range(len(self.params)))
        self.handle = None

    def copy_grads_in(self) -> None:
        for offset, numel, p in zip(self.offsets, self.numels, self.params):
            grad = p.grad if p.grad is not None else torch.zeros_like(p)
            self.buffer[offset : offset + numel].copy_(grad.reshape(-1))

    def copy_grads_out(self, world_size: int) -> None:
        self.buffer.div_(world_size)
        for offset, numel, p in zip(self.offsets, self.numels, self.params):
            if p.grad is None:
                p.grad = torch.empty_like(p)
            p.grad.copy_(self.buffer[offset : offset + numel].view_as(p))


def build_buckets(
    params: list[torch.nn.Parameter], bucket_size_mb: float = 25.0
) -> list[Bucket]:
    """Pack parameters into buckets of roughly `bucket_size_mb` each.

    Parameters are consumed in the order given. `DistributedDataParallel`
    passes them in reverse registration order, since that approximates
    the order in which their gradients become ready during backward
    (the last layer in the forward pass is the first to get a gradient).

    Assumes all parameters share one dtype; mixed-precision parameter
    groups (e.g. fp32 norms alongside bf16 linears) are not supported by
    this simplified bucketer.
    """
    bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
    buckets: list[Bucket] = []

    current_params: list[torch.nn.Parameter] = []
    current_offsets: list[int] = []
    current_numels: list[int] = []
    current_bytes = 0
    cursor = 0

    def flush() -> None:
        nonlocal current_params, current_offsets, current_numels, current_bytes, cursor
        if not current_params:
            return
        device = current_params[0].device
        dtype = current_params[0].dtype
        buffer = torch.zeros(cursor, dtype=dtype, device=device)
        bucket = Bucket(
            params=current_params,
            buffer=buffer,
            offsets=current_offsets,
            numels=current_numels,
        )
        bucket.reset()
        buckets.append(bucket)
        current_params, current_offsets, current_numels = [], [], []
        current_bytes = 0
        cursor = 0

    for p in params:
        if not p.requires_grad:
            continue
        numel = p.numel()
        nbytes = numel * p.element_size()
        if current_params and current_bytes + nbytes > bucket_size_bytes:
            flush()
        current_params.append(p)
        current_offsets.append(cursor)
        current_numels.append(numel)
        current_bytes += nbytes
        cursor += numel

    flush()
    return buckets
