"""Train a set of AutoGluon models using the given configuration."""

import datetime
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import dask.dataframe as dd
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import (
    get_cv_splits_dir,
    get_predict_imputed_fn,
    get_predict_mask_fn,
    get_trait_models_dir,
    get_y_fn,
    _configure_experiment_logging #CHARLIE EXPERIMENT EDIT - for setting up detailed logging in experiment runs
)
from src.utils.df_utils import pipe_log
from src.utils.log_utils import set_dry_run_text, suppress_dask_logging
from src.utils.training_utils import assign_weights, filter_trait_set


# CHARLIE EXPERIMENT EDIT
# change logging behaviour for detailed logging in experiment runs
import logging, io, contextlib
# enable wandb logging
import wandb
from src.utils.wandb_utils import wandb_log_metrics, wandb_log_table, wandb_log_dir_as_artifact, start_live_val_logger
import sys
import threading



class Tee:
    """Write to multiple streams (e.g. terminal + file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def fileno(self):
        for s in self.streams:
            if hasattr(s, "fileno"):
                return s.fileno()
        raise OSError("No underlying stream has a file descriptor")

    def isatty(self):
        for s in self.streams:
            if hasattr(s, "isatty"):
                try:
                    return s.isatty()
                except Exception:
                    pass
        return False

    @property
    def encoding(self):
        for s in self.streams:
            if hasattr(s, "encoding"):
                return s.encoding
        return "utf-8"





def _add_autogluon_file_handler(log_path: Path, level: int = logging.INFO) -> logging.FileHandler:
    """Attach a FileHandler to the 'autogluon' logger writing to log_path.
    Returns the handler so it can be removed (prevents duplicate logs across folds).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ag_logger = logging.getLogger("autogluon")
    ag_logger.setLevel(level)
    ag_logger.propagate = True

    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    ag_logger.addHandler(fh)
    return fh


def _remove_handler_safely(logger_name: str, handler: logging.Handler) -> None:
    try:
        lg = logging.getLogger(logger_name)
        lg.removeHandler(handler)
        handler.close()
    except Exception:
        pass

@dataclass
class TrainOptions:
    """Configuration for training AutoGluon models."""

    sample: float
    debug: bool
    resume: bool
    dry_run: bool
    cfg: ConfigBox = get_config()


@dataclass
class TraitSetInfo:
    """Configuration for training a single trait set for a single trait."""

    trait_set: str
    trait_name: str
    training_dir: Path
    cfg: ConfigBox = get_config()

    @property
    def cv_dir(self) -> Path:
        """Directory where cross-validation models are stored."""
        return self.training_dir / "cv"

    @property
    def cv_eval_results(self) -> Path:
        """Path to the cross-validation evaluation results."""
        return self.training_dir / self.cfg.train.eval_results

    @property
    def cv_feature_importance(self) -> Path:
        """Path to the cross-validation feature importance results."""
        return self.training_dir / self.cfg.train.feature_importance

    @property
    def full_model(self) -> Path:
        """Directory where the full model is stored."""
        return self.training_dir / "full_model"

    def cv_fold_complete_flag(self, fold: int) -> Path:
        """Flag to indicate if the cross-validation evaluation results are complete for
        a given fold."""
        return self.training_dir / "cv" / f"cv_fold_{fold}_complete.flag"

    def mark_cv_fold_complete(self, fold: int) -> None:
        """Mark the cross-validation evaluation results as complete for a given fold."""
        self.cv_fold_complete_flag(fold).touch()

    @property
    def cv_complete_flag(self) -> Path:
        """Flag to indicate if the cross-validation evaluation results are complete."""
        return self.training_dir / "cv_complete.flag"

    def mark_cv_complete(self) -> None:
        """Mark the cross-validation evaluation results as complete."""
        self.cv_complete_flag.touch()

    def full_model_complete_flag(self) -> Path:
        """Flag to indicate if the full model has been trained."""
        return self.training_dir / "full_model_complete.flag"

    def mark_full_model_complete(self) -> None:
        """Mark the full model as complete."""
        self.full_model_complete_flag().touch()

    @property
    def is_cv_complete(self) -> bool:
        """Check if the cross-validation evaluation results are complete."""
        return self.cv_complete_flag.exists()

    @property
    def is_full_model_complete(self) -> bool:
        """Check if the full model has been trained."""
        return self.full_model_complete_flag().exists()

    def get_last_complete_fold_id(self) -> int | None:
        """Get the ID of the last fold for which the cross-validation evaluation results
        are complete. If no folds are complete, return None."""
        complete_folds = list(self.cv_dir.glob("cv_fold_*_complete.flag"))
        if not complete_folds:
            return None

        return max(int(f.name.split("_")[2]) for f in complete_folds)


class TraitTrainer:
    """Train AutoGluon models for a single trait using the given configuration."""

    def __init__(self, xy: pd.DataFrame, trait_name: str, opts: TrainOptions):
        """Initialize the trait trainer."""
        self.xy = xy
        self.trait_name = trait_name
        self.opts = opts
        self.dry_run_text = set_dry_run_text(opts.dry_run)

        # CHARLIE EXPERIMENT EDIT
        # init wandb logging (fold-level + full-model-level)
        if wandb.run is not None:
            # Define trait-scoped metrics so multiple labels don't overwrite each other
            trait_prefix = f"{self.trait_name}"
            wandb.define_metric(f"{trait_prefix}/cv/fold")
            wandb.define_metric(f"{trait_prefix}/cv/*", step_metric=f"{trait_prefix}/cv/fold")

            # Keep TabM epoch metrics generic (the live logger expects these keys)
            wandb.define_metric("tabm_epoch/step")
            wandb.define_metric("tabm_epoch/val_score", step_metric="tabm_epoch/step")

            wandb.define_metric(f"{trait_prefix}/full_model/step")
            wandb.define_metric(f"{trait_prefix}/full_model/*", step_metric=f"{trait_prefix}/full_model/step")

        if opts.sample < 1.0:
            self._log_subsampling()
            self.xy = self._sample_xy() if not opts.dry_run else self.xy

        self.runs_dir: Path = (
            get_trait_models_dir(self.trait_name) / "debug"
            if opts.debug
            else get_trait_models_dir(self.trait_name)
        )

        sorted_runs = sorted(
            [run for run in self.runs_dir.glob("*") if "tmp" not in run.name],
            reverse=True,
        )

        if not sorted_runs:
            log.warning("No prior runs found in %s. Creating new run...", self.runs_dir)
            self.last_run: Path = self.runs_dir / now()
            self.current_run: Path = self.last_run
        else:
            self.last_run: Path = sorted_runs[0]
            self.current_run: Path = (
                self.last_run if opts.resume else self.runs_dir / now()
            )
        #CHARLIE EXPERIMENT EDIT
        # set up detailed autogluon logging to current run path
        self.current_run.mkdir(parents=True, exist_ok=True)
        _configure_experiment_logging(self.current_run)

    def _sample_xy(self) -> pd.DataFrame:
        """Sample the input data for quick prototyping."""
        return self.xy.sample(
            frac=self.opts.sample, random_state=self.opts.cfg.random_seed
        )

    def _log_is_trained_full(self, trait_set: str) -> None:
        """Log that all models (CV and full) have already been trained for the given
        trait set."""
        log.info(
            "All models for %s already trained for %s trait set. Skipping...%s",
            self.trait_name,
            trait_set,
            self.dry_run_text,
        )

    def _log_is_trained_cv(self, trait_set: str) -> None:
        """Log that CV models have already been trained for the given trait set."""
        log.info(
            "CV models for %s already trained for %s trait set. Skipping directly to "
            "full model training...%s",
            self.trait_name,
            trait_set.upper(),
            self.dry_run_text,
        )

    def _log_is_trained_partial_cv(self, trait_set: str) -> None:
        """Log that the CV model training is only partially complete for the given trait
        set."""
        log.info(
            "CV training for %s not complete for %s trait set. Resuming training...%s",
            self.trait_name,
            trait_set.upper(),
            self.dry_run_text,
        )

    def _log_training(self, trait_set: str) -> None:
        """Log that the model is being trained for the given trait set."""
        log.info(
            "Training model for %s with %s trait set...%s",
            self.trait_name,
            trait_set.upper(),
            self.dry_run_text,
        )

    def _log_subsampling(self) -> None:
        """Log that the data is being subsampled."""
        log.info(
            "Subsampling %i%% of the data...%s",
            self.opts.sample * 100,
            self.dry_run_text,
        )

    def _log_full_training(self, trait_set: str) -> None:
        """Log that the full model is being trained."""
        log.info(
            "Training model on all data for trait %s from %s trait set...%s",
            self.trait_name,
            trait_set.upper(),
            self.dry_run_text,
        )

    @staticmethod
    def _aggregate_results_original(cv_dir: Path, target: str) -> pd.DataFrame:
        return (
            pd.concat(
                [
                    pd.read_csv(fold_model_path / target, index_col=0)
                    for fold_model_path in cv_dir.glob("fold_*")
                ],
            )
            .drop(columns=["fold"])
            .reset_index(names="index")
            .groupby("index")
            .agg(["mean", "std"])
        )
    
    def _aggregate_results(self, cv_dir: Path, target: str | None):
        """Aggregate per-fold CSV results under `cv_dir` for the given `target` filename.

        Returns a DataFrame (or None) suitable for writing to CSV by callers.
        This handles both the per-fold evaluation CSVs (which contain a `fold` column)
        and feature-importance CSVs (which typically use the feature name as the index).
        """
        if not target:
            log.info("Aggregation skipped (no target configured).")
            return None

        rows = []
        for fold_model_path in sorted(cv_dir.glob("fold_*")):
            fp = fold_model_path / target
            if not fp.exists():
                log.warning("Result file missing, skipping: %s", fp)
                continue
            try:
                rows.append(pd.read_csv(fp, index_col=0))
            except Exception as e:
                log.exception("Failed to read result file %s: %s", fp, e)

        if not rows:
            log.warning("No result files found under %s for target %s", cv_dir, target)
            return None

        # Concatenate all per-fold frames
        try:
            concat = pd.concat(rows, axis=0, ignore_index=False)
        except Exception:
            concat = pd.concat(rows, axis=0, ignore_index=True)

        # If concatenated frame has a 'fold' column, treat as per-fold eval results
        if "fold" in concat.columns:
            # Set fold as index (one row per fold) and return
            try:
                result = concat.set_index("fold").sort_index()
            except Exception:
                result = concat
            return result

        # Otherwise assume feature-importance-like data (features as index) and aggregate
        try:
            result = concat.groupby(concat.index).agg(["mean", "std"])
        except Exception:
            result = concat

        return result


    # We set this to avoid a bug in LightGBM when used with GPU.
    # See https://github.com/microsoft/LightGBM/issues/3679

    
    def _train_full_model(self, ts_info: TraitSetInfo):
        ts_info.full_model.mkdir(parents=True, exist_ok=True)
        ag_log_path = ts_info.full_model / "autogluon.log"

        train_full = TabularDataset(
            self.xy.pipe(filter_trait_set, ts_info.trait_set)
            .dropna(subset=[self.trait_name])
            .pipe(assign_weights, w_gbif=self.opts.cfg.train.weights.gbif)
            .drop(columns=["x", "y", "source", "fold"])
        )

        HYPERPARAMS: dict = {
            "GBM": {"device": "cpu"},
            "TABM": {},
            "MITRA": {},
            "REALTABPFN-V2.5": {},
        }

        fit_kwargs = {
            "included_model_types": self.opts.cfg.autogluon.included_model_types,
            "num_gpus": self.opts.cfg.autogluon.num_gpus,
            "num_cpus": self.opts.cfg.autogluon.num_cpus,
            "presets": self.opts.cfg.autogluon.presets,
            "time_limit": self.opts.cfg.autogluon.full_fit_time_limit,
            "save_bag_folds": self.opts.cfg.autogluon.save_bag_folds,
            "hyperparameters": HYPERPARAMS,
            "feature_prune_kwargs": {},
            "verbosity": self.opts.cfg.autogluon.get("verbosity", 2),
            "num_bag_folds": self.opts.cfg.autogluon.get("num_bag_folds"),
            "num_bag_sets": self.opts.cfg.autogluon.get("num_bag_sets"),
            "num_stack_levels": self.opts.cfg.autogluon.get("num_stack_levels"),
            "dynamic_stacking": self.opts.cfg.autogluon.get("dynamic_stacking"),
            "auto_stack": self.opts.cfg.autogluon.get("auto_stack"),
            "ag_args_fit": self.opts.cfg.autogluon.get("ag_args_fit"),
            "keep_only_best": self.opts.cfg.autogluon.get("keep_only_best"),
            "save_space": self.opts.cfg.autogluon.get("save_space"),
            "refit_full": self.opts.cfg.autogluon.get("refit_full"),
            "set_best_to_refit_full": self.opts.cfg.autogluon.get("set_best_to_refit_full"),
        }
        fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

        # --- capture ALL AutoGluon output to file while keeping terminal output ---
        fh = _add_autogluon_file_handler(ag_log_path)
        try:
            with ag_log_path.open("a") as f:
                tee_out = Tee(sys.__stdout__, f)
                tee_err = Tee(sys.__stderr__, f)

                with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                    predictor = TabularPredictor(
                        label=ts_info.trait_name,
                        sample_weight="weights",
                        path=str(ts_info.full_model),
                    ).fit(train_full, **fit_kwargs)
        finally:
            _remove_handler_safely("autogluon", fh)

        # --- W&B: full model metrics + leaderboard + log file upload (consistent keys) ---
        if wandb.run is not None:
            meta = {
                "meta/trait": ts_info.trait_name,
                "meta/trait_set": ts_info.trait_set,
            }

            # There is no external holdout here; this is training-set eval only (label clearly).
            try:
                full_eval = predictor.evaluate(train_full, auxiliary_metrics=True, detailed_report=True)
            except Exception:
                full_eval = {}

            trait_prefix = f"{ts_info.trait_name}"
            metrics = {f"{trait_prefix}/full_model/step": 0, **meta}

            # Use stable keys (namespaced by trait) so W&B charts don't collide across labels
            for k, v in full_eval.items():
                if not isinstance(v, (int, float)):
                    continue
                if k == "root_mean_squared_error":
                    metrics[f"{trait_prefix}/full_model/rmse_train"] = float(v)
                else:
                    metrics[f"{trait_prefix}/full_model/{k}_train"] = float(v)

            wandb.log(metrics)

            # Leaderboard table (train_full) - log under a single top-level key
            try:
                lb = predictor.leaderboard(train_full, silent=True)
                if lb is not None and not lb.empty:
                    lb = lb.reset_index()
                    lb = lb.rename(columns={lb.columns[0]: 'model'})
                    lb = lb.assign(trait=ts_info.trait_name, trait_set=ts_info.trait_set)
                    wandb.log({"tables/leaderboard_full_model": wandb.Table(dataframe=lb)})
            except Exception:
                log.exception("Failed to log full-model leaderboard to W&B for %s", ts_info.trait_name)

            # Upload the log file as a namespaced file attachment
            try:
                wandb.log({f"logs/{ts_info.trait_name}/full_autogluon_log": wandb.File(str(ag_log_path))})
            except Exception:
                pass

        ts_info.mark_full_model_complete()
        predictor.save_space()

    def _train_fold(self, fold_id: int, cv_dir: Path, trait_set: str) -> None:
        log.info("Training model for fold %d...", fold_id)
        fold_model_path = cv_dir / f"fold_{fold_id}"
        fold_model_path.mkdir(parents=True, exist_ok=True)

       
        HYPERPARAMS: dict = {
            "GBM": {
                "device": "cpu",
                # "ag_args_fit": {
                #     "num_gpus": self.opts.cfg.autogluon.num_gpus // 4,
                #     "num_cpus": self.opts.cfg.autogluon.num_cpus // 4,
                # },
            },
            "TABM": {},
            "MITRA": {},
            "REALTABPFN-V2.5": {},
        }


        #CHARLIE EXPERIMENT EDIT
        # set up a logger for AutoGluon to capture its output during .fit()
        ag_logger = logging.getLogger("autogluon")
        ag_logger.setLevel(logging.INFO)

        # now with wandb
        ag_log_path = fold_model_path / "autogluon.log"

        train = TabularDataset(
            self.xy[self.xy["fold"] != fold_id]
            .pipe(filter_trait_set, trait_set)
            .dropna(subset=[self.trait_name])
            .pipe(assign_weights, w_gbif=self.opts.cfg.train.weights.gbif)
            .drop(columns=["x", "y", "source", "fold"])
            .reset_index(drop=True)
        )
        val = TabularDataset(
            self.xy[self.xy["fold"] == fold_id]
            .query("source == 's'")
            .dropna(subset=[self.trait_name])
            .assign(weights=1.0)
            .drop(columns=["x", "y", "source", "fold"])
            .reset_index(drop=True)
        )


        try:
            #CHARLIE EXPERIMENT EDIT
            fit_kwargs = {
            "included_model_types": self.opts.cfg.autogluon.included_model_types,
            "num_gpus": self.opts.cfg.autogluon.num_gpus,
            "num_cpus": self.opts.cfg.autogluon.num_cpus,
            "presets": self.opts.cfg.autogluon.presets,
            "time_limit": self.opts.cfg.autogluon.cv_fit_time_limit,
            "save_bag_folds": self.opts.cfg.autogluon.save_bag_folds,
            "hyperparameters": HYPERPARAMS,
            "feature_prune_kwargs": {},
            "verbosity": self.opts.cfg.autogluon.get("verbosity", 2),
            "num_bag_folds": self.opts.cfg.autogluon.get("num_bag_folds"),
            "num_bag_sets": self.opts.cfg.autogluon.get("num_bag_sets"),
            "num_stack_levels": self.opts.cfg.autogluon.get("num_stack_levels"),
            "dynamic_stacking": self.opts.cfg.autogluon.get("dynamic_stacking"),
            "auto_stack": self.opts.cfg.autogluon.get("auto_stack"),
            "ag_args_fit": self.opts.cfg.autogluon.get("ag_args_fit"),
            "keep_only_best": self.opts.cfg.autogluon.get("keep_only_best"),
            "save_space": self.opts.cfg.autogluon.get("save_space"),
            "refit_full": self.opts.cfg.autogluon.get("refit_full"),
            "set_best_to_refit_full": self.opts.cfg.autogluon.get("set_best_to_refit_full"),
            }
            # remove None entries (AutoGluon will use defaults)
            fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

            fh = _add_autogluon_file_handler(ag_log_path)
            try:
                # Start live val logging thread (tails autogluon.log)
                # Start live TabM "(val)" logging -> W&B (epoch axis)
                stop_event = threading.Event()
                t = None
                if wandb.run is not None:
                    # Log a fold-level step for this trait so metrics are namespaced
                    trait_prefix = f"{self.trait_name}"
                    wandb.log({
                        f"{trait_prefix}/cv/fold": int(fold_id),
                        "meta/trait": self.trait_name,
                        "meta/trait_set": trait_set,
                        "meta/fold": int(fold_id),
                    })
                    # Updated helper should log to tabm_epoch/* keys
                    t = start_live_val_logger(ag_log_path, stop_event=stop_event)

                with ag_log_path.open("a") as f:
                    # IMPORTANT: use __stdout__/__stderr so terminal stays alive even if W&B redirects sys.stdout
                    tee_out = Tee(sys.__stdout__, f)
                    tee_err = Tee(sys.__stderr__, f)

                    with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                        predictor = TabularPredictor(
                            label=self.trait_name,
                            sample_weight="weights",
                            path=str(fold_model_path),
                        ).fit(train, **fit_kwargs)

            finally:
                # stop live logger
                try:
                    if t is not None:
                        stop_event.set()
                except Exception:
                    pass

                _remove_handler_safely("autogluon", fh)
        
            fi_enabled = self.opts.cfg.autogluon.get("feature_importance", False)
            log.info("FI enabled: %r", fi_enabled)


            if fi_enabled:
                log.info("Calculating feature importance...")
                features = predictor.feature_metadata_in.get_features()
                feat_ds_map = {
                    "canopy_height": {"startswith": True, "match": "ETH"},
                    "soilgrids": {
                        "startswith": False,
                        "match": "cm_mean",
                    },
                    "modis": {"startswith": True, "match": "sur_refl"},
                    "vodca": {"startswith": True, "match": "vodca"},
                    "worldclim": {"startswith": True, "match": "wc2.1"},
                }
                # Generate a list of tuples of (dataset, [features]) for each dataset
                datasets = []
                for ds, ds_info in feat_ds_map.items():
                    if ds_info["startswith"]:
                        ds_feats = [
                            feat
                            for feat in features
                            if feat.startswith(ds_info["match"])
                        ]
                    else:
                        ds_feats = [
                            feat for feat in features if feat.endswith(ds_info["match"])
                        ]
                    datasets.append((ds, ds_feats))

                # Now add all features as well
                datasets += features

                feature_importance = predictor.feature_importance(
                    val,
                    features=datasets,
                    time_limit=self.opts.cfg.autogluon.FI_time_limit,
                    num_shuffle_sets=self.opts.cfg.autogluon.FI_num_shuffle_sets,
                ).assign(fold=fold_id)

                feature_importance.to_csv(
                    fold_model_path / self.opts.cfg.train.feature_importance
                )

            log.info(
                "Evaluating fold model (%s/%s)...",
                fold_id + 1,
                self.opts.cfg.train.cv_splits.n_splits,
            )
            eval_results = predictor.evaluate(
                val, auxiliary_metrics=True, detailed_report=True
            )

            # Normalize RMSE by the 99th percentile - 1st percentile range of the target
            norm_factor = val[self.trait_name].quantile(0.99) - val[
                self.trait_name
            ].quantile(0.01)
            eval_results["norm_root_mean_squared_error"] = (
                eval_results["root_mean_squared_error"] / norm_factor
            )

            #CHARLIE EXPERIMENT EDIT

            run = wandb.run
            if run is not None:
                    # Fold-level CV metrics (namespaced by trait to avoid collisions)
                    trait_prefix = f"{self.trait_name}"
                    metrics = {f"{trait_prefix}/cv/fold": int(fold_id)}

                    # Add metadata once per fold log
                    metrics.update({
                        "meta/trait": self.trait_name,
                        "meta/trait_set": trait_set,
                    })

                    for k, v in eval_results.items():
                        if isinstance(v, (int, float)):
                            # normalize naming a bit:
                            if k == "root_mean_squared_error":
                                metrics[f"{trait_prefix}/cv/rmse"] = float(v)
                            elif k == "norm_root_mean_squared_error":
                                metrics[f"{trait_prefix}/cv/nrmse"] = float(v)
                            else:
                                metrics[f"{trait_prefix}/cv/{k}"] = float(v)

                    wandb.log(metrics)

                    # Fold leaderboard table: log under a single top-level key
                    try:
                        lb = predictor.leaderboard(val, silent=True)
                        if lb is not None and not lb.empty:
                            lb = lb.reset_index()
                            lb = lb.rename(columns={lb.columns[0]: 'model'})
                            lb = lb.assign(trait=self.trait_name, trait_set=trait_set, fold=fold_id)
                            # Log to a single key to keep schema consistent across folds/traits
                            wandb.log({
                                "tables/leaderboard_fold": wandb.Table(dataframe=lb)
                            })
                    except Exception:
                        log.exception("Failed to log fold leaderboard to W&B for %s fold %s", self.trait_name, fold_id)

                    # Upload fold log file as a run artifact (namespaced key to avoid overwrite)
                    try:
                        wandb.log({f"logs/{self.trait_name}/fold_{fold_id}_autogluon_log": wandb.File(str(ag_log_path))})
                    except Exception:
                        pass

            pd.DataFrame({col: [val] for col, val in eval_results.items()}).assign(
                fold=fold_id
            ).to_csv(fold_model_path / self.opts.cfg.train.eval_results)

            predictor.save_space()

            # CHARLIE EXPERIMENT EDIT
            # deprecated
            """
            # Capture stdout/stderr produced by AutoGluon during .fit()
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                predictor = TabularPredictor(label=self.trait_name, path=fold_model_path).fit(
                    train_data=train, tuning_data=val, hyperparameters=HYPERPARAMS
                )
            # Forward captured outputs into the autogluon logger
            out = stdout_buf.getvalue().strip()
            err = stderr_buf.getvalue().strip()
            if out:
                ag_logger.info("AutoGluon stdout (fold %d):\n%s", fold_id, out)
            if err:
                ag_logger.error("AutoGluon stderr (fold %d):\n%s", fold_id, err)
            # also emit a structured message to your project logger
            log.info("Finished AutoGluon fit for fold %d, model saved to %s", fold_id, fold_model_path)
            """
   
   

        except ValueError as e:
            log.error("Error training model: %s", e)
            raise

    def _aggregate_cv_results(self, cv_dir: Path, training_dir: Path):
        log.info("Aggregating evaluation results...")
        eval_df = self._aggregate_results(cv_dir, self.opts.cfg.train.eval_results)
        eval_df.to_csv(training_dir / self.opts.cfg.train.eval_results)

        log.info("Aggregating feature importance...")
        fi_df = self._aggregate_results(cv_dir, self.opts.cfg.train.feature_importance)
        if fi_df is not None:
            fi_df.to_csv(training_dir / self.opts.cfg.train.feature_importance)
        else:
            log.info("No feature importance results to aggregate.")

    def _train_models_cv(self, ts_info: TraitSetInfo) -> None:
        ts_info.cv_dir.mkdir(parents=True, exist_ok=True)

        last_complete_fold = ts_info.get_last_complete_fold_id()
        starting_fold = last_complete_fold + 1 if last_complete_fold is not None else 0

        for i in range(starting_fold, max(self.xy["fold"].unique()) + 1):
            self._train_fold(i, ts_info.cv_dir, ts_info.trait_set)
            ts_info.mark_cv_fold_complete(i)

        self._aggregate_cv_results(ts_info.cv_dir, ts_info.training_dir)

        # --- W&B: aggregated CV results ---
        run = wandb.run
        if run is not None:
            eval_path = ts_info.training_dir / self.opts.cfg.train.eval_results
            if eval_path.exists():
                try:
                    # try simple CSV; if multi-index, fallback
                    try:
                        agg_df = pd.read_csv(eval_path)
                    except Exception:
                        agg_df = pd.read_csv(eval_path, header=[0, 1], index_col=0).reset_index()

                    # If agg_df is empty, skip logging
                    if agg_df is None or getattr(agg_df, "empty", True):
                        log.warning("Aggregated CV results exist but are empty: %s", eval_path)
                    else:
                        # Flatten any MultiIndex columns for W&B and ensure a flat index
                        if isinstance(agg_df.columns, pd.MultiIndex):
                            agg_df.columns = ["_".join(map(str, c)).strip() for c in agg_df.columns]
                        agg_df = agg_df.reset_index(drop=True)
                        wandb.log({
                            f"inner_cv_agg/{ts_info.trait_set}/{self.trait_name}": wandb.Table(dataframe=agg_df)
                        })
                except Exception:
                    log.exception("Failed to log aggregated CV results to W&B for %s/%s", ts_info.trait_set, self.trait_name)

        ts_info.mark_cv_complete()

    def _train_trait_set(self, trait_set: str) -> None:
        """Train AutoGluon models for a single trait using the given configuration."""
        dry_run = self.opts.dry_run

        ts_info = TraitSetInfo(
            trait_set,
            self.trait_name,
            self.current_run / trait_set,
        )

        if not dry_run:
            ts_info.training_dir.mkdir(parents=True, exist_ok=True)

        if ts_info.is_cv_complete and ts_info.is_full_model_complete:
            self._log_is_trained_full(trait_set)
            return

        if not ts_info.is_cv_complete:
            self._log_is_trained_partial_cv(trait_set)
            if not dry_run:
                self._train_models_cv(ts_info)
        else:
            self._log_is_trained_cv(trait_set)

        self._log_full_training(trait_set)
        if not dry_run:
            self._train_full_model(ts_info)

    def train_trait_models_all_y_sets(self) -> None:
        """Train a set of AutoGluon models for a single trait based on each trait set."""
        for trait_set in ["splot", "splot_gbif", "gbif"]:
            self._train_trait_set(trait_set)

    def train_splot(self) -> None:
        """Train AutoGluon models for the "splot" trait set."""
        self._train_trait_set("splot")

    def train_gbif(self) -> None:
        """Train AutoGluon models for the "gbif" trait set."""
        self._train_trait_set("gbif")

    def train_splot_gbif(self) -> None:
        """Train AutoGluon models for the "splot_gbif" trait set."""
        self._train_trait_set("splot_gbif")


def now() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def prep_full_xy(
    feats: dd.DataFrame,
    feats_mask: pd.DataFrame,
    labels: dd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    """
    Prepare the input data for modeling by filtering and assigning weights.

    Args:
        feats (dd.DataFrame): The input features.
        feats_mask (pd.DataFrame): The mask for filtering the features.
        labels (dd.DataFrame): The input labels.
        label_col (str): The column name of the labels.
        trait_set (str): The trait set to filter the data.

    Returns:
        pd.DataFrame: The prepared input data for modeling.
    """
    # TODO: #13 Speed up by leveraging dask for masking and merging
    log.info("Loading splits...")
    splits = (
        dd.read_parquet(get_cv_splits_dir() / f"{label_col}.parquet")
        .compute()
        .set_index(["y", "x"])
    )

    log.info("Merging splits and label data...")
    label = (
        labels[["x", "y", label_col, "source"]]
        .compute()
        .set_index(["y", "x"])
        .merge(splits, validate="m:1", right_index=True, left_index=True)
    )

    return (
        feats.compute()
        .set_index(["y", "x"])
        .pipe(pipe_log, "Masking features...")
        .mask(feats_mask)
        .pipe(pipe_log, "Merging features and label data...")
        .merge(label, validate="1:m", right_index=True, left_index=True)
        .reset_index()
    )


def load_data() -> tuple[dd.DataFrame, pd.DataFrame, dd.DataFrame]:
    """Load the input data for modeling."""
    feats = dd.read_parquet(get_predict_imputed_fn())
    feats_mask = pd.read_parquet(get_predict_mask_fn()).set_index(["y", "x"])
    labels = dd.read_parquet(get_y_fn())
    return feats, feats_mask, labels


def train_models(
    trait_sets: Iterable[str] | None = None,
    label_names: Iterable[str] | None = None,
    sample: float = 1.0,
    debug: bool = False,
    resume: bool = True,
    dry_run: bool = False,
    labels: list[str] | None = None,
) -> None:
    """Train a set of AutoGluon models for each  using the given configuration."""
    dry_run_text = set_dry_run_text(dry_run)
    suppress_dask_logging()

    valid_trait_sets = ("splot", "gbif", "splot_gbif")

    if trait_sets is None:
        trait_sets = valid_trait_sets

    train_opts = TrainOptions(sample, debug, resume, dry_run)

    log.info("Loading data...%s", dry_run_text)

    feats, feats_mask, labels = load_data()

    # CHARLIE EXPERIMENT EDIT
    # if labels given, filter
    # else train on all
    if label_names:
        label_cols = [col for col in label_names if col in labels.columns]
    else:
        label_cols = labels.columns.difference(["x", "y", "source"])


    for label_col in label_cols:
        tmp_xy_path = get_trait_models_dir(label_col) / "tmp" / "xy.parquet"

        if not tmp_xy_path.exists() and resume:
            runs = [
                run
                for run in get_trait_models_dir(label_col).glob("*")
                if run.is_dir() and "tmp" not in run.name
            ]

            if not runs:
                log.warning(
                    "No prior runs found for %s. Creating new run...%s",
                    label_col,
                    dry_run_text,
                )
            else:
                latest_run = max(
                    run
                    for run in get_trait_models_dir(label_col).glob("*")
                    if run.is_dir() and "tmp" not in run.name
                )

                completed = [
                    TraitSetInfo(
                        trait_set, label_col, latest_run / trait_set
                    ).is_full_model_complete
                    for trait_set in trait_sets
                ]

                if all(completed):
                    log.info(
                        "All models for %s already trained. Skipping...%s",
                        label_col,
                        dry_run_text,
                    )
                    continue

        log.info("Preparing data for %s training...%s", label_col, dry_run_text)
        if not tmp_xy_path.exists() or not resume:

            def _to_ddf(df: pd.DataFrame) -> dd.DataFrame:
                return dd.from_pandas(df, npartitions=100)

            if not dry_run:
                xy = prep_full_xy(feats, feats_mask, labels, label_col)
                log.info("Saving xy data for %s...%s", label_col, dry_run_text)
                tmp_xy_path.parent.mkdir(parents=True, exist_ok=True)
                xy.pipe(_to_ddf).to_parquet(
                    tmp_xy_path, compression="zstd", overwrite=True
                )
        else:
            log.info(
                "Found existing xy data for %s. Loading...%s", label_col, dry_run_text
            )
            if not dry_run:
                xy = dd.read_parquet(tmp_xy_path).compute().reset_index(drop=True)

        trait_trainer = TraitTrainer(xy, label_col, train_opts)
        for ts in trait_sets:
            if ts not in valid_trait_sets:
                raise ValueError(f"Invalid trait set: {ts}")
            if ts == "splot":
                trait_trainer.train_splot()
            elif ts == "gbif":
                trait_trainer.train_gbif()
            elif ts == "splot_gbif":
                trait_trainer.train_splot_gbif()

        log.info("Cleaning up...%s", dry_run_text)
        if not dry_run:
            shutil.rmtree(tmp_xy_path.parent)

    log.info("Done! \U00002705")
