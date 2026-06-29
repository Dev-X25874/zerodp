"""torchrun --standalone --nproc-per-node <N> examples/train_fsdp.py

Trains TinyGPT using the from-scratch FSDP-style parameter sharding in
zerodp.fsdp. Every transformer block gets its own FSDPUnit; everything
else (embeddings, final norm, output head) is swept up by
auto_wrap_children so no parameter is left unsharded -- and therefore
unsynchronised -- by accident.
"""
import torch
import torch.nn.functional as F

from model import TinyGPT
from zerodp.comm import barrier, get_rank, get_world_size, init_distributed
from zerodp.fsdp import auto_wrap_children, wrap_module_list


def make_batch(batch_size: int, block_size: int, vocab_size: int, device: torch.device, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, vocab_size, (batch_size, block_size + 1), generator=g)
    idx = idx.to(device)
    return idx[:, :-1], idx[:, 1:]


def main() -> None:
    device = init_distributed()
    rank, world_size = get_rank(), get_world_size()

    vocab_size, block_size = 256, 128
    model = TinyGPT(vocab_size=vocab_size, block_size=block_size).to(device)

    model.blocks = wrap_module_list(model.blocks)
    auto_wrap_children(model, skip=("blocks",))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(50):
        x, y = make_batch(
            batch_size=16,
            block_size=block_size,
            vocab_size=vocab_size,
            device=device,
            seed=step * world_size + rank,
        )
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if rank == 0 and step % 10 == 0:
            print(f"step {step:3d} | loss {loss.item():.4f}")

    barrier()
    if rank == 0:
        print("done.")


if __name__ == "__main__":
    main()
