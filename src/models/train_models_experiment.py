import argparse
from typing import Iterable, Any, Optional

from box import ConfigBox

import yaml
import json
import datetime
from pathlib import Path

from src.conf.conf import get_config, update_config
from src.conf.environment import activate_env


import logging
from logging.handlers import RotatingFileHandler


import wandb
from src.utils.wandb_utils import wandb_init_if_enabled, wandb_log_yaml



def parse_trait_sets(t: str | None, cfg: dict| None) -> list[str]:
    if t:
        return [x.strip() for x in t.split(",") if x.strip()]
    return list(cfg["trait_sets"])

def parse_labels(s: str | None, cfg: dict) -> list[str]:
    if s:
        if s=="all":
            return None
        else:
            return [x.strip() for x in s.split(",") if x.strip()]
    if cfg["labels"]:
        if cfg["labels"]=="all":
            return None
        else: return list(cfg["labels"])
    else: return None

# Safely extract expected fields with sensible defaults.
def _get(o, name, default):
    try:
        return getattr(o, name)
    except Exception:
        try:
            return o.get(name, default)
        except Exception:
            return default

def _cfg_to_primitive(obj: Any) -> Any:
    """
    Recursively convert ConfigBox / Box / nested objects to plain Python types
    so yaml.safe_dump can serialize them.
    """
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return _cfg_to_primitive(obj.to_dict())
    if isinstance(obj, dict):
        return {k: _cfg_to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cfg_to_primitive(v) for v in obj]
    return obj



def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a subset of trait models")
    p.add_argument(
        "-l",
        "--labels",
        help="Comma-separated label names to train (e.g. X14_mean). Default from config",
    )
    p.add_argument(
        "-t",
        "--trait-sets",
        help="Comma-separated trait sets to train (splot,gbif,splot_gbif). Default from config",
    )
    p.add_argument(
        "--config-dir",
        default="experiments/configs",
        help="Directory containing experiment configs (yml/json)",
    )
    p.add_argument(
        "--experiment-name",
        required=True,
        help="Name of the experiment config file (without extension) to load from --config-dir",
    )
    #p.add_argument("-s", "--sample", type=float, default=1.0, help="Sample fraction")
    #p.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    #p.add_argument("-r", "--resume", action="store_true", help="Resume")
    #p.add_argument("-n", "--dry-run", action="store_true", help="Dry run")
    return p.parse_args()




def main(args: argparse.Namespace, cfg: ConfigBox = get_config()) -> None:
    # keep same environment activation as train_models
    activate_env()

    # load experiment config file (yaml or json)
    cfg_dir = Path(args.config_dir)
    exp_name = args.experiment_name
    exp_path = None
    for ext in ("yml", "yaml", "json"):
        p = cfg_dir / f"{exp_name}.{ext}"
        if p.exists():
            exp_path = p
            break
    if exp_path is None:
        raise SystemExit(f"Experiment config not found for '{exp_name}' in {cfg_dir}")

    if exp_path.suffix in (".yml", ".yaml"):
        with exp_path.open("r") as fh:
            exp_cfg = yaml.safe_load(fh)
    else:
        with exp_path.open("r") as fh:
            exp_cfg = json.load(fh)

     # new save dir logic
    save_dir = f"{exp_cfg['experiment']['save_dir']}/{exp_cfg['experiment']['name']}"  #{old_model_dir}"
    exp_cfg["models"] = {}
    exp_cfg["models"]["dir"] = save_dir

     # pick from CLI args if provided, else config
    trait_sets = parse_trait_sets(args.trait_sets, exp_cfg)
    labels = parse_labels(args.labels, exp_cfg)

    # Override the actual cfg used downstream
    exp_cfg["trait_sets"] = trait_sets
    exp_cfg["labels"] = labels

    update_config(exp_cfg or {})
     # re-fetch the (now updated) global cfg so we are operating on the same object everywhere
    cfg = get_config()


    # delayed import to ensure config updates are in place before modules that rely on it are imported
    from src.models import autogluon
   
    save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)


    # getting train options
    if hasattr(cfg, "opts"):
        opts_src = getattr(cfg, "opts")
    else: opts_src=None


    print(f"Saving experiment config used for this run to {save_dir}")

    used_cfg_path = save_dir / "experiment_config_used.yml"
    with used_cfg_path.open("w") as fh:
        yaml.safe_dump(_cfg_to_primitive(cfg), fh)
  

    run = None

    try:
   
        # initialize wandb
        run = wandb_init_if_enabled(cfg, run_dir=save_dir, exp_name=exp_name)
        if run is not None:
            wandb_log_yaml(run, key="full_config", yaml_path=used_cfg_path)
            (save_dir / "wandb_run_id.txt").write_text(run.id)

        # Delay importing training entrypoint until after config & logging are set
        #import autogluon
        from src.models.autogluon import train_models

        print(cfg.autogluon.included_model_types)
        

        # Call the autogluon training entrypoint so resume/run logic is respected globally
        train_models(
            trait_sets=trait_sets,
            label_names=labels,
            dry_run=_get(opts_src, "dry_run", False),
            sample=_get(opts_src, "sample", 1.0),
            debug=_get(opts_src, "debug", False),
            resume=_get(opts_src, "resume", False),
        )

        print("Done.")

    finally:
        if run is not None:
            wandb.finish()



if __name__ == "__main__":
    main(cli())