#!/usr/bin/env python3
"""
Custom script to run spatial cross-validation splits specifically for Shrub Tree Grass 55km data.

This script is based on the main skcv_splits.py but is tailored for:
- PFT: Shrub_Tree_Grass  
- Model resolution: 55km
- Custom output directory for the splits

Usage:
    python run_shrub_tree_grass_55km_splits.py [--output-dir OUTPUT_DIR] [--overwrite] [--debug]
"""

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Optional

# Set up environment variables before importing project modules
if "PROJECT_ROOT" not in os.environ:
    # Assume we're running from the project root
    project_root = Path(__file__).parent.absolute()
    os.environ["PROJECT_ROOT"] = str(project_root)
    
# Add project root to Python path
import sys
if os.environ["PROJECT_ROOT"] not in sys.path:
    sys.path.insert(0, os.environ["PROJECT_ROOT"])

# Create necessary directories
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

import dask.dataframe as dd
import pandas as pd
from box import ConfigBox

# Import project modules (avoid importing skcv_splits to prevent cli() function conflict)
from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import get_autocorr_ranges_fn
from src.utils.log_utils import get_loggers_starting_with


def assign_folds_iteration(
    df: pd.DataFrame, n_folds: int, trait_col: str
) -> tuple[pd.DataFrame, float]:
    """Assign folds to a DataFrame and calculate similarity."""
    from collections.abc import Sequence
    from scipy.stats import ks_2samp
    
    def calculate_kg_p_value(
        df: pd.DataFrame, data_col: str, fold_i: int, fold_j: int
    ) -> float:
        """Calculate the p-value using the Kolmogorov-Smirnov test for two folds."""
        folds_df = df[df["fold"].isin([fold_i, fold_j])]
        folds_values = folds_df[data_col]
        mask = folds_df["fold"] == fold_i
        fold_i_values = folds_values[mask]
        fold_j_values = folds_values[~mask]
        _, p_value = ks_2samp(fold_i_values, fold_j_values)
        return p_value

    def calculate_similarity_kg(folds: Sequence, df: pd.DataFrame, data_col: str) -> float:
        """Calculate the similarity between folds using the Kolmogorov-Smirnov test."""
        n_comparisons = len(folds) * (len(folds) - 1) // 2
        if n_comparisons == 0:
            return 0.0
        
        total_p_value = 0.0
        for i, fold_i in enumerate(folds):
            for fold_j in folds[i + 1:]:
                p_value = calculate_kg_p_value(df, data_col, fold_i, fold_j)
                total_p_value += p_value
        
        return total_p_value / n_comparisons

    from src.utils.spatial_utils import assign_hexagons
    
    # Group by hexagon and assign folds
    hex_folds = (
        df.groupby("hex")
        .size()
        .reset_index(name="count")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    hex_folds["fold"] = hex_folds.index % n_folds
    
    # Merge back with original DataFrame
    df_with_folds = df.merge(hex_folds[["hex", "fold"]], on="hex")
    
    # Calculate similarity
    folds = list(range(n_folds))
    similarity = calculate_similarity_kg(folds, df_with_folds, trait_col)
    
    return df_with_folds, similarity


def assign_folds(
    df: pd.DataFrame, n_folds: int, n_sims: int, trait_col: str
) -> pd.DataFrame:
    """Assign folds to a DataFrame using the best simulation."""
    log.info(f"Running {n_sims} simulations to find best fold assignment...")
    
    best_df = None
    best_similarity = -1
    
    for sim in range(n_sims):
        try:
            df_sim, similarity = assign_folds_iteration(df, n_folds, trait_col)
            if similarity > best_similarity:
                best_similarity = similarity
                best_df = df_sim.copy()
        except Exception as e:
            log.warning(f"Simulation {sim} failed: {e}")
            continue
    
    if best_df is None:
        raise ValueError("All simulations failed")
    
    log.info(f"Best similarity score: {best_similarity:.4f}")
    return best_df


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate spatial k-fold cross-validation splits for Shrub Tree Grass 55km data."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/features/Shrub_Tree_Grass/55km/custom_skcv_splits",
        help="Output directory for the splits (default: data/features/Shrub_Tree_Grass/55km/custom_skcv_splits)"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite existing splits"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    return parser.parse_args()


def get_custom_y_fn() -> Path:
    """Get the path to the Y.parquet file for Shrub Tree Grass 55km."""
    return Path("data/features/Shrub_Tree_Grass/55km/Y.parquet")


def get_custom_autocorr_ranges_fn() -> Optional[Path]:
    """
    Get the path to the autocorrelation ranges file for Shrub Tree Grass.
    
    Returns None if no suitable file is found.
    """
    # Try different possible autocorrelation range files
    possible_files = [
        "reference/spatial_autocorr_Shrub_Tree_Grass_55km_mean.parquet",
        "reference/spatial_autocorr_Shrub_Tree_Grass_22km_mean.parquet", 
        "reference/spatial_autocorr_Shrub_Tree_Grass_1km_mean.parquet",
        "reference/spatial_autocorr_1km.parquet",
        "reference/spatial_autocorr_001.parquet"
    ]
    
    for file_path in possible_files:
        path = Path(file_path)
        if path.exists():
            log.info(f"Using autocorrelation ranges file: {path}")
            return path
    
    log.warning("No autocorrelation ranges file found. Will try to use default from config.")
    return None


def create_custom_config() -> ConfigBox:
    """Create a custom configuration for Shrub Tree Grass 55km."""
    cfg = get_config()
    
    # Override configuration for our specific use case
    custom_cfg = cfg.copy()
    custom_cfg.PFT = "Shrub_Tree_Grass"
    custom_cfg.model_res = "55km"
    
    # Set target resolution for 55km (approximately 55000 meters)
    custom_cfg.target_resolution = 55000
    
    return custom_cfg


def custom_assign_trait_splits(
    traits_df: dd.DataFrame,
    trait_col: str,
    ranges: pd.DataFrame,
    overwrite: bool,
    cfg: ConfigBox,
    custom_output_dir: Path,
) -> None:
    """
    Modified version of _assign_trait_splits that saves to custom directory.
    """
    custom_output_dir.mkdir(parents=True, exist_ok=True)
    splits_fn = custom_output_dir / f"{trait_col}.parquet"

    if splits_fn.exists() and not overwrite:
        log.info("Splits for trait %s already exist. Skipping...", trait_col)
        return

    log.info("Processing trait: %s", trait_col)
    # Ensure dask loggers don't interfere with the main logger
    dask_loggers = get_loggers_starting_with("distributed")
    for logger in dask_loggers:
        logging.getLogger(logger).setLevel(logging.WARNING)

    trait_df = traits_df[[trait_col, "x", "y"]].dropna()

    # Try to get trait range from the ranges DataFrame
    trait_range_rows = ranges[ranges["trait"] == trait_col]
    if len(trait_range_rows) > 0:
        trait_range = trait_range_rows[cfg.train.cv_splits.range_stat].values[0]
    else:
        # Fallback: use the target resolution if trait not found in ranges
        log.warning(f"Trait {trait_col} not found in ranges. Using target resolution.")
        trait_range = cfg.target_resolution

    # Rest of the processing follows the original logic from _assign_trait_splits
    from src.utils.spatial_utils import acr_to_h3_res, assign_hexagons
    from src.utils.df_utils import reproject_xy_to_geo
    
    if cfg.crs == "EPSG:4326":
        trait_range_deg = trait_range / 111320
        if trait_range_deg <= cfg.target_resolution:
            log.warning(
                "Trait range of %.2f m is less than or equal to the existing map "
                "resolution of %.2f m. Using the map resolution for hexagon assignment...",
                trait_range,
                cfg.target_resolution,
            )
            trait_range = cfg.target_resolution * 111320

    if cfg.crs == "EPSG:6933":
        if trait_range <= cfg.target_resolution:
            log.warning(
                "Trait range of %.2f m is less than or equal to the existing map "
                "resolution of %.2f m. Using the map resolution for hexagon assignment...",
                trait_range,
                cfg.target_resolution,
            )
            trait_range = cfg.target_resolution

        log.info("Reprojecting coordinates to WGS84 for hexagon assignment...")
        meta = {
            trait_col: "float64",
            "x": "float64",
            "y": "float64",
            "lon": "float64",
            "lat": "float64",
        }
        trait_df = trait_df.map_partitions(
            reproject_xy_to_geo, from_crs=cfg.crs, meta=meta
        )
        trait_df = trait_df.rename(
            columns={"x": "x_old", "y": "y_old", "lat": "y", "lon": "x"}
        )

    h3_res = acr_to_h3_res(trait_range)
    trait_df = assign_hexagons(trait_df, h3_res, dask=True).reset_index(drop=True)

    if cfg.crs == "EPSG:6933":
        # Revert back to the original coordinates
        trait_df = trait_df.drop(columns=["x", "y"]).rename(
            columns={"x_old": "x", "y_old": "y"}
        )

    if isinstance(trait_df, dd.DataFrame):
        log.info("Computing trait dask DataFrame...")
        trait_df = trait_df.compute()

    log.info("Assigning the best folds...")
    trait_df = assign_folds(
        trait_df,
        cfg.train.cv_splits.n_splits,
        cfg.train.cv_splits.n_sims,
        trait_col,
    )

    # Save to custom output directory
    trait_df[["x", "y", "fold"]].drop_duplicates(subset=["x", "y"]).reset_index(
        drop=True
    ).to_parquet(splits_fn, compression="zstd")
    
    log.info(f"Saved splits for {trait_col} to {splits_fn}")


def main(args: argparse.Namespace) -> None:
    """Main function to generate spatial k-fold cross-validation splits."""

    print("In main")
    
    # Create custom configuration
    cfg = create_custom_config()
    syscfg = cfg[detect_system()][cfg.model_res]
    
    # Ignore warnings
    warnings.simplefilter(action="ignore", category=UserWarning)
    
    if args.debug:
        log.setLevel(logging.DEBUG)
        # Use fewer workers in debug mode
        syscfg.skcv_splits.n_workers = min(10, syscfg.skcv_splits.n_workers)

    log.info("Initializing Dask...")
    client, _ = init_dask(
        dashboard_address=cfg.dask_dashboard,
        n_workers=syscfg.skcv_splits.n_workers,
    )
    
    # Try to load autocorrelation ranges
    ranges_path = get_custom_autocorr_ranges_fn()
    if ranges_path:
        ranges = pd.read_parquet(
            ranges_path,
            columns=["trait", cfg.train.cv_splits.range_stat],
        )
    else:
        # Create a dummy ranges dataframe if none found
        log.warning("No ranges file found, creating dummy ranges.")
        ranges = pd.DataFrame({
            "trait": ["dummy"],
            cfg.train.cv_splits.range_stat: [cfg.target_resolution]
        })

    # Load the Y data
    y_path = get_custom_y_fn()
    if not y_path.exists():
        log.error(f"Y data file not found at {y_path}")
        close_dask(client)
        return
        
    log.info(f"Loading Y data from {y_path}")
    target_cols = dd.read_parquet(y_path).columns.difference(["source"])
    trait_cols = target_cols.difference(["x", "y"])

    traits = dd.read_parquet(y_path, columns=target_cols).repartition(
        npartitions=100
    )

    # Create custom output directory
    custom_output_dir = Path(args.output_dir)
    log.info(f"Saving splits to: {custom_output_dir}")
    
    log.info("Assigning splits for traits...")
    log.info(f"Found {len(trait_cols)} traits to process: {list(trait_cols)}")
    
    # Process each trait
    for trait_col in trait_cols:
        try:
            custom_assign_trait_splits(
                traits, trait_col, ranges, args.overwrite, cfg, custom_output_dir
            )
        except Exception as e:
            log.error(f"Failed to process trait {trait_col}: {e}")
            if args.debug:
                raise  # Re-raise in debug mode for full traceback
            continue

    close_dask(client)
    log.info("Done!")
    log.info(f"Splits saved to: {custom_output_dir}")


if __name__ == "__main__":
    args = cli()
    main(args)