#!/usr/bin/env bash
set -euo pipefail

datasets=(
  lastfm
  mimicii_jitter
  retweet_jitter
  stackoverflow
)

usage() {
  echo "Usage: $0 [all|DATASET]" >&2
  echo "Datasets: ${datasets[*]}" >&2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

selection="${1:-all}"
if [[ "${selection}" == "all" ]]; then
  selected=("${datasets[@]}")
else
  selected=()
  for dataset in "${datasets[@]}"; do
    if [[ "${selection}" == "${dataset}" ]]; then
      selected=("${dataset}")
      break
    fi
  done
  if [[ "${#selected[@]}" -eq 0 ]]; then
    echo "Unknown dataset: ${selection}" >&2
    usage
    exit 2
  fi
fi

cd "$(dirname "$0")/.."

for dataset in "${selected[@]}"; do
  echo "[NJDTPP] training ${dataset}"
  python examples/train_nhp.py \
    --config_dir examples/configs/exp_config_njdtpp.yaml \
    --experiment_id "NJDTPP_${dataset}_train"
done
