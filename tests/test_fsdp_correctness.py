"""Correctness check for zerodp.fsdp.FSDPUnit.

Two simulated ranks (gloo, CPU, via mp.spawn) each hold half of a
TinyMLP's parameters as a flat shard. Concatenating the two ranks'
gradient shards back together should reproduce exactly the gradient a
single unsharded process gets from the mean loss over the concatenation
of both ranks' batches.
"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from zerodp.fsdp import FSDPUnit


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
    sharded = FSDPUnit(base_model)

    x = torch.full((4, 8), float(rank + 1))
    loss = sharded(x).pow(2).mean()
    loss.backward()

    torch.save(sharded.local_shard.grad.clone(), f"{results_path}.rank{rank}.pt")
    dist.destroy_process_group()


def test_fsdp_shard_grads_match_single_process_reference(tmp_path):
    world_size = 2
    init_method = f"file://{tmp_path / 'rdvz'}"
    results_path = str(tmp_path / "shardgrads")

    mp.spawn(_worker, args=(world_size, init_method, results_path), nprocs=world_size, join=True)
    shard_grads = [torch.load(f"{results_path}.rank{r}.pt") for r in range(world_size)]
    flat_shard_grad = torch.cat(shard_grads)

    torch.manual_seed(0)
    ref_model = TinyMLP()
    x = torch.cat([torch.full((4, 8), float(r + 1)) for r in range(world_size)], dim=0)
    loss = ref_model(x).pow(2).mean()
    loss.backward()
    flat_ref_grad = torch.cat([p.grad.reshape(-1) for p in ref_model.parameters()])

    assert torch.allclose(flat_shard_grad[: flat_ref_grad.numel()], flat_ref_grad, atol=1e-6)
