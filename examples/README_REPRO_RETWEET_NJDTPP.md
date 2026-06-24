# Retweet NJDTPP Paper Default

Run the paper/source default directly through EasyTPP:

```bash
./examples/run_retweet_njdtpp_paper_default.sh
```

This script uses `examples/configs/train_retweet_njdtpp_paper_default.yaml`.

For a short timing check, use the optional grid runner with one trial and one
epoch:

```bash
python examples/run_repro_sweep.py \
  --config_dir examples/configs/train_retweet_njdtpp_paper_default.yaml \
  --max_trials 1 \
  --max_runs 1 \
  --max_epoch 1 \
  --gpu 0 \
  --sweep_name retweet_njdtpp_timing
```

Set `trainer_config.gpu` in the YAML to your GPU id, or use the optional grid
runner override:

```bash
python examples/run_repro_sweep.py \
  --config_dir examples/configs/train_retweet_njdtpp_paper_default.yaml \
  --grid examples/configs/repro_retweet_njdtpp_grid.yaml \
  --gpu 0 \
  --sweep_name retweet_njdtpp_grid
```

The grid runner is only a helper.  It writes generated configs under
`checkpoints/repro_sweeps/<sweep_name>/generated_configs/` and skips trials
that already have `state/<trial_id>/done.json`.
