from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml
from datetime import datetime
import wandb

import re
import threading
import time

_VAL_RE = re.compile(r"\(val\)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_EPOCH_RE = re.compile(r"epoch['\"]?\s*:\s*(\d+)")



def start_live_val_logger(
    log_path: Path,
    stop_event: threading.Event,
    poll_s: float = 0.25,
) -> threading.Thread:
    """
    Tail `log_path` and log TabM internal validation scores to W&B as a clean time-series.

    Logs ONLY numeric metrics (no meta fields) to avoid chart issues:
      - tabm_epoch/step (int)     : increments on every matched '(val) ...' line
      - tabm_epoch/val_score (float): the parsed validation score

    Requirements:
      - W&B run must already be initialized (wandb.run is not None).
      - Define metrics once somewhere before training starts:
            wandb.define_metric("tabm_epoch/step")
            wandb.define_metric("tabm_epoch/val_score", step_metric="tabm_epoch/step")
    """
    def _worker():
        if wandb.run is None:
            return

        step = 0

        # Wait briefly for file to exist (race: thread starts before file open)
        for _ in range(40):  # ~10s max
            if stop_event.is_set():
                return
            if log_path.exists():
                break
            time.sleep(0.25)

        try:
            with open(log_path, "r") as f:
                # Start at end so we only capture new lines produced during this fit
                f.seek(0, 2)

                while not stop_event.is_set():
                    line = f.readline()
                    if not line:
                        time.sleep(poll_s)
                        continue

                    m = _VAL_RE.search(line)
                    if not m:
                        continue

                    try:
                        val = float(m.group(1))
                    except ValueError:
                        continue

                    step += 1
                    wandb.log({
                        "tabm_epoch/step": step,
                        "tabm_epoch/val_score": val,
                    })

        except FileNotFoundError:
            # If the file disappears or never existed, just stop quietly
            return
        except Exception:
            # Avoid killing training because the logging thread died
            return

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t

def wandb_init_if_enabled(cfg: Any, run_dir: Path, exp_name: str, settings = wandb.Settings(console="wrap")) -> Optional[wandb.sdk.wandb_run.Run]:
    """
    Initialize W&B if enabled in config. Returns the run or None.
    Assumes cfg can be converted to primitives elsewhere (you already have _cfg_to_primitive in train_models_experiment.py).
    """
    enabled = False
    try:
        enabled = bool(cfg.get("wandb", {}).get("enabled", False))
    except Exception:
        enabled = False

    if not enabled:
        return None

    project = cfg.wandb.get("project", os.getenv("WANDB_PROJECT", "geosense"))
    entity = cfg.wandb.get("entity", os.getenv("WANDB_ENTITY"))
    tags = list(cfg.wandb.get("tags", [])) if hasattr(cfg, "wandb") else []

    # --- Attach simplified timestamp ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"{exp_name}-{timestamp}"

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=exp_name,           # group by experiment name for easy comparison in W&B UI
        dir=str(run_dir),            # ensures wandb files land inside your experiment folder
        tags=tags,
        reinit=True,
        settings=settings
            )
    return run


def wandb_log_yaml(run, key: str, yaml_path: Path) -> None:
    if run is None or not yaml_path.exists():
        return
    run.save(str(yaml_path))  # uploads as a file artifact-like entry
    with yaml_path.open("r") as f:
        data = yaml.safe_load(f)
    # best effort: put it into config too (flattening is optional; leaving nested is ok)
    run.config.update({key: data}, allow_val_change=True)


def wandb_log_metrics(run, metrics: Dict[str, float], step: int | None = None) -> None:
    if run is None:
        return
    wandb.log(metrics, step=step)


def wandb_log_table(run, name: str, df) -> None:
    if run is None:
        return
    run.log({name: wandb.Table(dataframe=df)})


def wandb_log_dir_as_artifact(run, artifact_name: str, dir_path: Path, artifact_type: str = "run_dir") -> None:
    if run is None or not dir_path.exists():
        return
    art = wandb.Artifact(artifact_name, type=artifact_type)
    art.add_dir(str(dir_path))
    run.log_artifact(art)