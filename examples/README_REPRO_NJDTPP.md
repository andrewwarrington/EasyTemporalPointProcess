# NJDTPP Paper-Default Runs

Standalone NJDTPP repro entrypoint. This does not depend on the Decoupled
config or README.

Config:

```text
examples/configs/exp_config_njdtpp.yaml
```

Defaults:

- Model: hidden size 32, two hidden layers, tanh coefficient networks,
  initialization std 0.01.
- SDE: 10 Euler-Maruyama subdivisions between events.
- Likelihood grid: 11 points, matching the 10 sub-intervals plus endpoints
  used by the released code.
- Optimizer: Adam lr 1e-3, eta0 lr 1e-1, weight decay 1e-5.
- Training: batch size 30 and seed 42 from the authors' released scripts.
- Run budget: EasyTPP uses epochs, while the released NJDTPP scripts use 3000
  random minibatch iterations, so `max_epoch` is intentionally just the simple
  EasyTPP budget knob.

Run all configured datasets:

```bash
./examples/run_njdtpp_paper_defaults.sh
```

Run one dataset:

```bash
./examples/run_njdtpp_paper_defaults.sh retweet_jitter
```

Available datasets:

```text
lastfm mimicii_jitter retweet_jitter stackoverflow
```

Sources:

- Paper: https://proceedings.mlr.press/v235/zhang24cm.html
- Released code: https://github.com/Zh-Shuai/NJDTPP
