"""Correctness check for zerodp.ddp.DistributedDataParallel.

Two simulated ranks (gloo, CPU, via mp.spawn) each backward a different
4-sample batch through the same initial weights. After
finish_gradient_synchronization, every rank should hold the identical,
averaged gradient -- and that gradient should equal the gradient a
single unsharded process gets from the mean loss over the concatenation
of both ranks' batches (true because grad of mean over a combined batch
equals the average of the two per-half means' gradients).
"""
import copy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from zerodp.ddp import DistributedDataParallel


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _worker(rank: int, world_size: int, init_method: str, results_path: str) -> None:
    dist.init_process_group(
        backend="gloo", init_method=init_method, rank=rank, world_size=world_size
    )
    torch.manual_seed(0)
    base_model = TinyMLP()

    model = DistributedDataParallel(copy.deepcopy(base_model))
    x = torch.full((4, 8), float(rank + 1))
    loss = model(x).pow(2).mean()
    loss.backward()
    model.finish_gradient_synchronization()

    grads = {name: p.grad.clone() for name, p in model.module.named_parameters()}
    torch.save(grads, f"{results_path}.rank{rank}.pt")
    dist.destroy_process_group()


def test_ddp_ranks_agree_after_sync(tmp_path):
    world_size = 2
    init_method = f"file://{tmp_path / 'rdvz1'}"
    results_path = str(tmp_path / "grads")

    mp.spawn(_worker, args=(world_size, init_method, results_path), nprocs=world_size, join=True)

    grads_per_rank = [torch.load(f"{results_path}.rank{r}.pt") for r in range(world_size)]
    for name in grads_per_rank[0]:
        averaged = sum(g[name] for g in grads_per_rank) / world_size
        for g in grads_per_rank:
            assert torch.allclose(g[name], averaged, atol=1e-6), name


def test_ddp_matches_single_process_reference(tmp_path):
    world_size = 2
    init_method = f"file://{tmp_path / 'rdvz2'}"
    results_path = str(tmp_path / "grads2")

    mp.spawn(_worker, args=(world_size, init_method, results_path), nprocs=world_size, join=True)
    distributed_grads = torch.load(f"{results_path}.rank0.pt")

    torch.manual_seed(0)
    ref_model = TinyMLP()
    x = torch.cat([torch.full((4, 8), float(r + 1)) for r in range(world_size)], dim=0)
    loss = ref_model(x).pow(2).mean()
    loss.backward()

    for name, p in ref_model.named_parameters():
        assert torch.allclose(p.grad, distributed_grads[name], atol=1e-6), name
