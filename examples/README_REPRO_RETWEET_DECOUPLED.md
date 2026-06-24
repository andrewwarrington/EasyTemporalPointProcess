# Retweet Decoupled Paper Default

Run the paper default directly through EasyTPP:

```bash
./examples/run_retweet_decoupled_paper_default.sh
```

For a short timing check, temporarily lower `trainer_config.max_epoch` in the
YAML or use the optional grid runner override.

Optional grid:

```bash
python examples/run_repro_sweep.py \
  --config_dir examples/configs/train_retweet_decoupled_paper_default.yaml \
  --grid examples/configs/repro_retweet_decoupled_grid.yaml \
  --gpu 0 \
  --sweep_name retweet_decoupled_grid
```

The grid runner is only a helper.  It writes generated configs under
`checkpoints/repro_sweeps/<sweep_name>/generated_configs/` and skips trials
that already have `state/<trial_id>/done.json`.
