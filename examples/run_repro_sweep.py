"""Simple optional EasyTPP grid runner.

For one paper-default run, use examples/train_nhp.py directly with a config.
Use this script only when you want a small Cartesian grid.  It is re-entrant:
completed trials are skipped, and a tiny lock directory prevents two workers
from taking the same trial at the same time.
"""

from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import logging
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
LOGGER = logging.getLogger("repro_sweep")

TRAINER_FIELDS = {
    "batch_size",
    "gpu",
    "learning_rate",
    "max_epoch",
    "seed",
    "weight_decay",
}

MODEL_FIELDS = {
    "dropout_rate",
    "hidden_size",
    "loss_integral_num_sample_per_step",
    "num_layers",
    "use_mc_samples",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments for the optional grid runner.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_dir", "--config-dir", type=Path, required=True)
    parser.add_argument("--experiment_id", "--experiment-id", default=None)
    parser.add_argument("--grid", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2019])
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output_root", "--output-root", type=Path, default=Path("checkpoints/repro_sweeps"))
    parser.add_argument("--sweep_name", "--sweep-name", default=None)
    parser.add_argument("--max_trials", "--max-trials", type=int, default=None)
    parser.add_argument("--max_runs", "--max-runs", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--max_epoch", "--max-epoch", type=int, default=None)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=None)
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--retry_failed", "--retry-failed", action="store_true")
    parser.add_argument("--stop_on_error", "--stop-on-error", action="store_true")
    parser.add_argument("--log_level", "--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level_name: str) -> None:
    """Configure this script's logger.

    Args:
        level_name: Logging level name such as ``INFO`` or ``DEBUG``.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOGGER.setLevel(level)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML mapping, or an empty mapping for an empty file.
    """
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a dictionary as YAML.

    Args:
        path: Destination path.
        payload: YAML-serializable mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically.

    Args:
        path: Destination path.
        payload: JSON-serializable mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(tmp_path, path)


def infer_experiment_id(config: dict[str, Any], requested: str | None) -> str:
    """Resolve the EasyTPP experiment id to run.

    Args:
        config: Parsed EasyTPP YAML config.
        requested: Explicit experiment id from the CLI, if any.

    Returns:
        Experiment id.

    Raises:
        ValueError: If the id cannot be inferred from a multi-experiment config.
    """
    if requested:
        return requested
    ids = [
        key
        for key in config
        if key not in {"pipeline_config_id", "data"} and not key.startswith("_")
    ]
    if len(ids) != 1:
        raise ValueError("Pass --experiment_id when a config has multiple experiments.")
    return ids[0]


def parse_value(raw: str) -> tuple[str, Any]:
    """Parse a ``KEY=VALUE`` override.

    Args:
        raw: Raw CLI override string.

    Returns:
        Pair of key and YAML-parsed value.

    Raises:
        ValueError: If the string is not shaped like ``KEY=VALUE``.
    """
    key, sep, value = raw.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"Expected KEY=VALUE, got {raw!r}")
    return key.strip(), yaml.safe_load(value)


def normalize_path(key: str) -> str:
    """Map shorthand override keys to EasyTPP config paths.

    Args:
        key: Full dotted path or shorthand key.

    Returns:
        Dotted config path under ``trainer_config``, ``model_config``, or
        ``model_config.model_specs``.
    """
    if key.startswith("model_specs."):
        return f"model_config.{key}"
    if "." in key:
        return key
    if key in TRAINER_FIELDS:
        return f"trainer_config.{key}"
    if key in MODEL_FIELDS:
        return f"model_config.{key}"
    return f"model_config.model_specs.{key}"


def set_path(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a nested dictionary value by dotted path.

    Args:
        payload: Mapping to mutate.
        dotted_path: Dot-separated key path.
        value: Value to write.
    """
    current = payload
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def load_grid(path: Path | None) -> list[dict[str, Any]]:
    """Load and expand a Cartesian grid file.

    Args:
        path: Optional YAML file with either a top-level ``grid`` mapping or a
            direct mapping of config paths to value lists.

    Returns:
        List of parameter dictionaries, one per grid point. If no grid is
        provided, returns one empty parameter set.
    """
    if path is None:
        return [{}]
    payload = read_yaml(path)
    grid = payload.get("grid", payload)
    keys = list(grid)
    values = [value if isinstance(value, list) else [value] for value in grid.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def build_trial_config(
    base: dict[str, Any],
    experiment_id: str,
    params: dict[str, Any],
    seed: int,
    index: int,
    sweep_dir: Path,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Create the generated config for one grid trial.

    Args:
        base: Parsed base EasyTPP config.
        experiment_id: Base experiment id in ``base``.
        params: Grid and constant parameter overrides.
        seed: Trial seed.
        index: One-based trial index.
        sweep_dir: Root directory for this sweep.
        args: Parsed CLI arguments.

    Returns:
        Tuple of generated trial id, generated EasyTPP YAML payload, and the
        normalized applied parameter mapping.
    """
    trial_id = f"{experiment_id}_seed{seed}_trial{index:04d}"
    experiment = copy.deepcopy(base[experiment_id])
    generated = {
        "pipeline_config_id": base["pipeline_config_id"],
        "data": copy.deepcopy(base.get("data", {})),
    }

    applied = {}
    for raw_key, value in params.items():
        key = normalize_path(raw_key)
        set_path(experiment, key, value)
        applied[key] = value

    trainer = experiment.setdefault("trainer_config", {})
    trainer["seed"] = seed
    if args.gpu is not None:
        trainer["gpu"] = args.gpu
    if args.max_epoch is not None:
        trainer["max_epoch"] = args.max_epoch
    if args.batch_size is not None:
        trainer["batch_size"] = args.batch_size

    experiment.setdefault("base_config", {})["base_dir"] = str(sweep_dir / "runs" / trial_id)
    experiment["base_config"].setdefault("specs", {})["sweep_params"] = applied
    generated[trial_id] = experiment
    return trial_id, generated, applied


def claim(state_dir: Path) -> bool:
    """Claim a trial with an atomic lock directory.

    Args:
        state_dir: Trial state directory.

    Returns:
        ``True`` if this process claimed the trial, otherwise ``False``.
    """
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        lock_dir = state_dir / "lock"
        lock_dir.mkdir()
    except FileExistsError:
        return False
    write_json(
        lock_dir / "owner.json",
        {"host": socket.gethostname(), "pid": os.getpid(), "time": time.time()},
    )
    return True


def release(state_dir: Path) -> None:
    """Release a trial lock if this process owns a normal lock directory.

    Args:
        state_dir: Trial state directory.
    """
    lock_dir = state_dir / "lock"
    try:
        (lock_dir / "owner.json").unlink()
        lock_dir.rmdir()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def close_easy_tpp_log(log_path: str | None) -> None:
    """Close the EasyTPP file handler for one trial.

    Args:
        log_path: Path of the EasyTPP log file handler to remove.
    """
    if not log_path:
        return
    try:
        from easy_tpp.utils import logger
    except ImportError:
        return
    expected = os.path.abspath(log_path)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == expected:
            logger.removeHandler(handler)
            handler.close()


def run_easy_tpp(config_path: Path, trial_id: str) -> dict[str, Any]:
    """Run one generated EasyTPP trial.

    Args:
        config_path: Generated YAML config path.
        trial_id: Experiment id inside the generated YAML.

    Returns:
        Result metadata with output paths and best metrics when available.
    """
    from easy_tpp.config_factory import Config
    from easy_tpp.runner import Runner

    config = None
    runner = None
    log_path = None
    try:
        config = Config.build_from_yaml_file(str(config_path), experiment_id=trial_id)
        runner = Runner.build_from_config(config)
        log_path = config.base_config.specs.get("saved_log_dir")
        runner.run()
        result = {
            "status": "done",
            "log_folder": config.base_config.specs.get("log_folder"),
            "saved_model": config.base_config.specs.get("saved_model_dir"),
            "resolved_config": config.base_config.specs.get("output_config_dir"),
        }
        if hasattr(runner, "metrics_tracker"):
            result["best_metrics"] = runner.metrics_tracker.current_best
            result["best_epoch"] = runner.metrics_tracker.episode_best
        return result
    finally:
        if runner is not None:
            del runner
        if config is not None:
            log_path = log_path or config.base_config.specs.get("saved_log_dir")
            del config
        gc.collect()
        close_easy_tpp_log(log_path)


def main() -> None:
    """Run the optional grid workflow."""
    args = parse_args()
    setup_logging(args.log_level)
    base = read_yaml(args.config_dir)
    experiment_id = infer_experiment_id(base, args.experiment_id)
    grid = load_grid(args.grid)
    constants = dict(parse_value(raw) for raw in args.sets)

    sweep_name = args.sweep_name or f"{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir = args.output_root / sweep_name
    config_dir = sweep_dir / "generated_configs"
    results_dir = sweep_dir / "results"

    trials = [(params | constants, seed) for params in grid for seed in args.seeds]
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    LOGGER.info("Experiment: %s", experiment_id)
    LOGGER.info("Trials: %d", len(trials))
    LOGGER.info("Sweep dir: %s", sweep_dir)

    runs = 0
    for index, (params, seed) in enumerate(trials, start=1):
        trial_id, trial_config, applied = build_trial_config(
            base, experiment_id, params, seed, index, sweep_dir, args
        )
        state_dir = sweep_dir / "state" / trial_id
        done_path = state_dir / "done.json"
        failed_path = state_dir / "failed.json"
        config_path = config_dir / f"{trial_id}.yaml"

        if args.dry_run:
            write_yaml(config_path, trial_config)
            LOGGER.info("[dry-run] %s: %s", trial_id, applied)
            continue
        if done_path.exists() or (failed_path.exists() and not args.retry_failed):
            LOGGER.info("[skip] %s", trial_id)
            continue
        if not claim(state_dir):
            LOGGER.info("[locked] %s", trial_id)
            continue

        write_yaml(config_path, trial_config)
        LOGGER.info("[run] %s", trial_id)
        try:
            result = run_easy_tpp(config_path, trial_id)
            result_path = results_dir / f"{trial_id}.json"
            write_json(result_path, result)
            write_json(done_path, {"trial_id": trial_id, "result": str(result_path), "params": applied})
            LOGGER.info("[done] %s", trial_id)
        except Exception as exc:
            write_json(failed_path, {"trial_id": trial_id, "error": repr(exc), "params": applied})
            LOGGER.exception("[failed] %s", trial_id)
            if args.stop_on_error:
                raise
        finally:
            release(state_dir)

        runs += 1
        if args.max_runs is not None and runs >= args.max_runs:
            break


if __name__ == "__main__":
    main()
