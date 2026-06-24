# Decoupled Paper-Default Runs

Standalone Decoupled/Dec-ODE repro entrypoint. This does not depend on the
NJDTPP config or README.

Config:

```text
examples/configs/exp_config_decoupled.yaml
```

Defaults:

- Influence state dimension `D`: 64.
- ODE network width `N`: 256.
- ODE network layers `L`: 3.
- Training solver: Euler with 16 fixed steps between events.
- Validation/testing solver: RK4 with 64 fixed steps between events.
- Trainer budget: the paper appendix does not specify a universal batch size
  or epoch count, so those remain plain EasyTPP knobs in the YAML.

Run all configured datasets:

```bash
./examples/run_decoupled_paper_defaults.sh
```

Run one dataset:

```bash
./examples/run_decoupled_paper_defaults.sh retweet_jitter
```

Available datasets:

```text
lastfm mimicii_jitter retweet_jitter stackoverflow
```

Source:

- Paper: https://arxiv.org/abs/2406.06149
