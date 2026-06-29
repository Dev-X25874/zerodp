#!/usr/bin/env bash
# Launch the DDP example on a single node with NPROC GPUs (default 2).
set -euo pipefail
NPROC="${1:-2}"
torchrun --standalone --nproc-per-node "$NPROC" examples/train_ddp.py
