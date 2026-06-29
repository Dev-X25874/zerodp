#!/usr/bin/env bash
# Launch the FSDP example on a single node with NPROC GPUs (default 2).
set -euo pipefail
NPROC="${1:-2}"
torchrun --standalone --nproc-per-node "$NPROC" examples/train_fsdp.py
