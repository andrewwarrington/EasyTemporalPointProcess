#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python examples/train_nhp.py \
  --config_dir examples/configs/train_retweet_decoupled_paper_default.yaml \
  --experiment_id Decoupled_retweet_paper_default
