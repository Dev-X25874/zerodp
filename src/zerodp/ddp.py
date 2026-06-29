"""A from-scratch reimplementation of `torch.nn.parallel.DistributedDataParallel`.

Wraps an `nn.Module`, broadcasts its initial parameters from rank 0, and
overlaps gradient all-reduce with backward computation via fixed-size
buckets (see `bucket.py`). Gradients are averaged (not summed) across
ranks, matching upstream DDP's semantics, so loss scaling does not need
to change when going from a single GPU to many.

This is a reference implementation, not a drop-in replacement for the
real thing. It reproduces the core mechanism -- bucketed, overlapped
all-reduce, triggered by per-parameter gradient-ready hooks -- without
the production hardening: no static-graph fast path, no SyncBatchNorm
integration, and no handling for parameters that go unused on some
forward passes but not others.
"""
from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from .bucket import Bucket, build_buckets
from .comm import broadcast_module, get_world_size, is_dist_available_and_initialized


class DistributedDataParallel(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.world_size = get_world_size()
        self._sync_enabled = True

        if is_dist_available_and_initialized() and self.world_size > 1:
            broadcast_module(self.module, src=0)

        # Reverse registration order approximates backward-readiness order.
        params = [p for p in reversed(list(self.module.parameters())) if p.requires_grad]
        self.buckets: list[Bucket] = build_buckets(params, bucket_size_mb=bucket_size_mb)

        self._param_to_location: dict[int, tuple[int, int]] = {}
        for bucket_idx, bucket in enumerate(self.buckets):
            for param_idx, p in enumerate(bucket.params):
                self._param_to_location[id(p)] = (bucket_idx, param_idx)
                p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, param: torch.nn.Parameter):
        def hook(p: torch.nn.Parameter) -> None:
            if not self._sync_enabled or self.world_size == 1:
                return
            bucket_idx, param_idx = self._param_to_location[id(p)]
            bucket = self.buckets[bucket_idx]
            if bucket.mark_ready(param_idx):
                bucket.copy_grads_in()
                bucket.handle = dist.all_reduce(
                    bucket.buffer, op=dist.ReduceOp.SUM, async_op=True
                )

        return hook

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Block on every in-flight all-reduce and write averaged grads back.

        Call this once after `loss.backward()` and before `optimizer.step()`.
        Real DDP does this automatically via an autograd graph callback;
        this implementation keeps it as an explicit call for clarity.
        """
        if self.world_size == 1:
            return
        for bucket in self.buckets:
            if bucket.handle is not None:
                bucket.handle.wait()
                bucket.copy_grads_out(self.world_size)
            bucket.reset()

    @contextlib.contextmanager
    def no_sync(self):
        """Disable gradient synchronisation for the duration of the block.

        Useful for gradient accumulation: run N-1 backward passes inside
        `no_sync()` and only let the final microbatch's backward trigger
        all-reduce.
        """
        previous = self._sync_enabled
        self._sync_enabled = False
        try:
            yield
        finally:
            self._sync_enabled = previous

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.load_state_dict(*args, **kwargs)
