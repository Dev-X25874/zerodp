# zerodp

From-scratch DDP gradient bucketing and FSDP-style ZeRO-3 parameter sharding for PyTorch, built directly on `torch.distributed` collectives. No `torch.nn.parallel.DistributedDataParallel`, no `torch.distributed.fsdp` -- just NCCL/gloo collectives, a couple of `autograd.Function`s, and parameter hooks.

## What's here

- `src/zerodp/ddp.py` -- `DistributedDataParallel`: bucketed, overlapped gradient all-reduce. Parameters are packed into ~25MB buckets in reverse-registration order; a per-parameter `register_post_accumulate_grad_hook` fires an async `all_reduce` the moment every gradient in a bucket has landed, so communication overlaps with the rest of backward instead of waiting for it to finish.
- `src/zerodp/fsdp.py` -- `FSDPUnit`: each wrapped submodule flattens and evenly shards its own parameters across ranks. A custom `autograd.Function` (`_AllGatherFlat`) all-gathers the full parameter on the forward pass and reduce-scatters (sum, then divide by world size) the gradient on the backward pass -- the sharding is otherwise invisible to autograd.
- `src/zerodp/bucket.py`, `src/zerodp/comm.py` -- the bucketing data structure and the thin `torch.distributed` wrappers both of the above are built on.
- `examples/` -- a 6-layer TinyGPT trained with each wrapper (`train_ddp.py`, `train_fsdp.py`), plus `model.py`.
- `tests/` -- correctness tests that spawn 2 simulated ranks over gloo on CPU (no GPU required) and check gradients against a single-process, unsharded reference -- see "Why the tests are actually meaningful" below.

## Install

```bash
pip install -e ".[dev]"
```

## Run the examples

```bash
./scripts/launch_ddp.sh 2    # or: torchrun --standalone --nproc-per-node 2 examples/train_ddp.py
./scripts/launch_fsdp.sh 2
```

Single-GPU / CPU also works -- both wrappers degrade to a no-op (`world_size == 1`) when there's no process group to talk to.

## Run the tests

```bash
pytest tests/ -v
```

## Why the tests are actually meaningful

For a loss that's a mean over a batch, the gradient of the mean over a *combined* batch equals the average of the gradients of the means over each *half* of that batch:
