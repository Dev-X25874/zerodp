"""Unit tests for the bucketing logic, independent of any process group."""
import torch
import torch.nn as nn

from zerodp.bucket import build_buckets


def test_build_buckets_packs_all_params_and_preserves_numel():
    params = [nn.Parameter(torch.zeros(1000)) for _ in range(5)]
    bytes_per_param = 1000 * 4  # float32
    buckets = build_buckets(params, bucket_size_mb=bytes_per_param * 2.5 / (1024 * 1024))

    assert sum(len(b.params) for b in buckets) == len(params)
    total_numel = sum(p.numel() for b in buckets for p in b.params)
    assert total_numel == sum(p.numel() for p in params)
    # forced to split: 2.5 params' worth of budget per bucket, 5 params total
    assert len(buckets) > 1


def test_build_buckets_skips_frozen_params():
    trainable = nn.Parameter(torch.zeros(10))
    frozen = nn.Parameter(torch.zeros(10), requires_grad=False)
    buckets = build_buckets([trainable, frozen], bucket_size_mb=10.0)
    all_params = [p for b in buckets for p in b.params]
    assert any(p is trainable for p in all_params)
    assert not any(p is frozen for p in all_params)


def test_bucket_mark_ready_triggers_only_once_all_present():
    params = [nn.Parameter(torch.zeros(4)) for _ in range(3)]
    bucket = build_buckets(params, bucket_size_mb=10.0)[0]
    assert bucket.mark_ready(0) is False
    assert bucket.mark_ready(1) is False
    assert bucket.mark_ready(2) is True


def test_bucket_copy_in_and_out_round_trips_average():
    p0 = nn.Parameter(torch.zeros(3))
    p1 = nn.Parameter(torch.zeros(2))
    bucket = build_buckets([p0, p1], bucket_size_mb=10.0)[0]

    p0.grad = torch.tensor([1.0, 2.0, 3.0])
    p1.grad = torch.tensor([4.0, 5.0])
    bucket.copy_grads_in()

    # simulate a 2-rank all-reduce(sum) by doubling the buffer in place
    bucket.buffer.mul_(2)
    bucket.copy_grads_out(world_size=2)

    assert torch.allclose(p0.grad, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(p1.grad, torch.tensor([4.0, 5.0]))
