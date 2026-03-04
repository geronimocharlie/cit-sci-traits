"""Split the data into train and test sets using spatial k-fold cross-validation."""

import argparse
import logging
import warnings
from collections.abc import Sequence

import dask.dataframe as dd
import numpy as np
import numpy.typing as npt
import pandas as pd
from box import ConfigBox
from dask import compute, delayed
from scipy.stats import ks_2samp

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import get_autocorr_ranges_fn, get_cv_splits_dir, get_y_fn
from src.utils.df_utils import reproject_xy_to_geo
from src.utils.log_utils import get_loggers_starting_with
from src.utils.spatial_utils import acr_to_h3_res, assign_hexagons


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate spatial k-fold cross-validation splits."
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing splits"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Main function to generate spatial k-fold cross-validation splits."""
    
    # Add custom log file handler
    import os
    charlie_log = logging.FileHandler("log/charlie.log", mode="w")
    charlie_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    charlie_log.setFormatter(formatter)
    log.addHandler(charlie_log)
    
    log.info("=" * 80)
    log.info("STARTING SPATIAL CROSS-VALIDATION SPLITS GENERATION")
    log.info("=" * 80)
    
    syscfg = cfg[detect_system()][cfg.model_res]
    
    log.info("Configuration:")
    log.info("- System: %s", detect_system())
    log.info("- PFT (Plant Functional Type): %s", cfg.PFT)
    log.info("- Model resolution: %s", cfg.model_res) 
    log.info("- Target resolution: %s meters", cfg.target_resolution)
    log.info("- CRS: %s", cfg.crs)
    log.info("- CV folds: %d", cfg.train.cv_splits.n_splits)
    log.info("- CV simulations: %d", cfg.train.cv_splits.n_sims)
    log.info("- Range statistic: %s", cfg.train.cv_splits.range_stat)

    # Ignore warnings
    warnings.simplefilter(action="ignore", category=UserWarning)

    if args.debug:
        log.setLevel(logging.DEBUG)
        syscfg.skcv_splits.n_workers = 40

    log.info("Initializing Dask cluster...")
    log.info("- Workers: %d", syscfg.skcv_splits.n_workers)
    log.info("- Dashboard: %s", cfg.dask_dashboard)
    client, _ = init_dask(
        dashboard_address=cfg.dask_dashboard,
        n_workers=syscfg.skcv_splits.n_workers,
        # threads_per_worker=syscfg.skcv_splits.threads_per_worker,
    )
    
    # Load spatial autocorrelation ranges
    ranges_file = get_autocorr_ranges_fn()
    log.info("Loading spatial autocorrelation ranges from: %s", ranges_file)
    ranges = pd.read_parquet(
        ranges_file,
        columns=["trait", cfg.train.cv_splits.range_stat],
    )
    log.info("Loaded ranges for %d traits", len(ranges))
    log.info("Range statistics available: %s", list(ranges.columns))
    log.info("Sample of ranges data:")
    log.info("\n%s", ranges.head(10).to_string())

    # Load trait data
    y_file = get_y_fn()
    log.info("Loading trait data from: %s", y_file)
    
    # First, peek at the full structure
    all_cols = dd.read_parquet(y_file).columns
    log.info("All columns in Y data: %s", list(all_cols))
    
    # Identify target columns (everything except 'source' if it exists)
    target_cols: pd.Index = all_cols.difference(["source"])
    log.info("Target columns (excluding 'source'): %s", list(target_cols))
    
    # Identify trait columns (target columns minus coordinates)
    trait_cols: pd.Index = target_cols.difference(["x", "y"])
    log.info("Trait columns (excluding x, y coordinates): %s", list(trait_cols))
    log.info("Number of traits to process: %d", len(trait_cols))

    # Load the actual data
    traits = dd.read_parquet(y_file, columns=target_cols).repartition(
        npartitions=100
    )
    
    # Log data structure information
    log.info("Trait dataset information:")
    log.info("- Partitions: %d", traits.npartitions)
    log.info("- Columns: %s", list(traits.columns))
    
    # Check for 'source' column to understand data sources
    if 'source' in all_cols:
        source_sample = dd.read_parquet(y_file, columns=['source']).head(1000)
        unique_sources = source_sample['source'].unique()
        log.info("Data sources found: %s", list(unique_sources))
    else:
        log.info("No 'source' column found in the data")

    log.info("Starting trait processing loop...")
    log.info("Processing %d traits: %s", len(trait_cols), list(trait_cols))
    
    # Process each trait
    for i, trait_col in enumerate(trait_cols, 1):
        log.info("Processing trait %d/%d: %s", i, len(trait_cols), trait_col)
        try:
            _assign_trait_splits(traits, trait_col, ranges, args.overwrite, cfg)
            log.info("Successfully completed trait %s", trait_col)
        except Exception as e:
            log.error("Failed to process trait %s: %s", trait_col, str(e))
            if args.debug:
                log.exception("Full traceback:")
            continue

    close_dask(client)
    log.info("=" * 80)
    log.info("SPATIAL CROSS-VALIDATION SPLITS GENERATION COMPLETED")
    log.info("=" * 80)


def calculate_kg_p_value(
    df: pd.DataFrame, data_col: str, fold_i: int, fold_j: int
) -> float:
    """
    Calculate the p-value using the Kolmogorov-Smirnov test for two folds in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        data_col (str): The column name of the data to compare.
        fold_i (int): The index of the first fold.
        fold_j (int): The index of the second fold.

    Returns:
        float: The p-value calculated using the Kolmogorov-Smirnov test.
    """
    folds_df = df[df["fold"].isin([fold_i, fold_j])]
    folds_values = folds_df[data_col]
    mask = folds_df["fold"] == fold_i
    fold_i_values = folds_values[mask]
    fold_j_values = folds_values[~mask]
    _, p_value = ks_2samp(fold_i_values, fold_j_values)
    return p_value  # pyright: ignore[reportReturnType]


def calculate_similarity_kg(folds: Sequence, df: pd.DataFrame, data_col: str) -> float:
    """
    Calculate the similarity between folds using the Kolmogorov-Smirnov test.

    Parameters:
    - folds (Sequence): A sequence of folds.
    - df (pd.DataFrame): The DataFrame containing the data.
    - data_col (str): The name of the column containing the data.

    Returns:
    - float: The similarity between the folds based on the Kolmogorov-Smirnov test.
    """

    # Calculate the pairwise comparisons
    p_values = [
        calculate_kg_p_value(df, data_col, folds[i], folds[j])
        for i in range(len(folds))
        for j in range(i + 1, len(folds))
    ]

    # Return the mean p-value as the similarity score
    return float(np.mean(p_values))


def assign_folds_iteration(
    df: pd.DataFrame, n_folds: int, data_col: str, hexagons: npt.NDArray
) -> tuple[float, pd.Series]:
    """
    Assigns folds to the hexagons in the given dataframe based on the number of folds
    specified.

    Parameters:
    - df: The input dataframe containing the hexagon data.
    - n_folds: The number of folds to assign.
    - data_col: The column name in the dataframe containing the data.
    - hexagons: The array of hexagons to assign folds to.

    Returns:
    - A tuple containing the similarity score and a copy of the fold assignments.
    """
    np.random.shuffle(hexagons)
    folds = np.array_split(hexagons, n_folds)
    hexagon_to_fold = {hexagon: i for i, fold in enumerate(folds) for hexagon in fold}
    df["fold"] = df["hex_id"].map(hexagon_to_fold)

    similarity = calculate_similarity_kg(range(n_folds), df, data_col)
    return similarity, df["fold"].copy()


def assign_folds(
    df: pd.DataFrame, n_folds: int, n_iterations: int, data_col: str
) -> pd.DataFrame:
    """
    Assigns folds to the given DataFrame based on similarity scores.

    Args:
        df (pd.DataFrame): The DataFrame to assign folds to.
        n_folds (int): The number of folds to assign.
        n_iterations (int): The number of iterations to perform.
        data_col (str): The column name in the DataFrame containing the data.

    Returns:
        pd.DataFrame: The DataFrame with the folds assigned.

    """
    hexagons = df["hex_id"].unique()

    results = compute(
        *[
            delayed(assign_folds_iteration)(df, n_folds, data_col, hexagons)
            for _ in range(n_iterations)
        ]
    )

    def _compute_best_similarity(
        results: list[tuple[float, pd.Series]],
    ) -> tuple[pd.Series, float]:
        best_similarity = None
        best_assignment = pd.Series(dtype=int)
        
        for similarity, assignment in results:
            if best_similarity is None:
                log.info("Similarity: %e. Current best: None", similarity)
            else:
                log.info("Similarity: %e. Current best: %e", similarity, best_similarity)
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_assignment = assignment
        
        if best_similarity is None:
            raise ValueError("No best similarity found.")

        return best_assignment, best_similarity

    best_assignment, best_similarity = _compute_best_similarity(results)
    log.info("Best similarity: %e", best_similarity)
    df["fold"] = best_assignment.astype(int)

    return df


def get_splits(
    df: pd.DataFrame,
) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """
    Generate train-test splits based on the 'fold' column in the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data and the 'fold' column.

    Returns:
        list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]: A list of tuples,
            where each tuple contains the train and test indices for a fold.
    """
    splits = []
    folds = df["fold"].unique()
    for fold in folds:
        train = df[df["fold"] != fold].index.to_numpy()
        test = df[df["fold"] == fold].index.to_numpy()
        splits.append((train, test))
    return splits


def _assign_trait_splits(
    traits_df: dd.DataFrame,
    trait_col: str,
    ranges: pd.DataFrame,
    overwrite: bool,
    cfg: ConfigBox,
) -> None:
    """
    Assign spatial cross-validation folds to trait data.
    
    This function:
    1. Loads trait data for a specific trait
    2. Finds the spatial autocorrelation range for that trait
    3. Creates hexagonal spatial clusters based on the range
    4. Assigns CV folds to clusters to minimize spatial autocorrelation
    5. Saves the fold assignments to a parquet file
    """
    splits_dir = get_cv_splits_dir()
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_fn = splits_dir / f"{trait_col}.parquet"

    if splits_fn.exists() and not overwrite:
        log.info("Splits for trait %s already exist. Skipping...", trait_col)
        return

    log.info("=" * 60)
    log.info("Processing trait: %s", trait_col)
    log.info("Output file: %s", splits_fn)
    
    # Ensure dask loggers don't interfere with the main logger
    dask_loggers = get_loggers_starting_with("distributed")
    for logger in dask_loggers:
        logging.getLogger(logger).setLevel(logging.WARNING)

    # Step 1: Extract trait data (trait values + coordinates)
    # We need x, y coordinates for spatial clustering and trait values for fold balancing
    log.info("Step 1: Extracting trait data for %s", trait_col)
    trait_df = traits_df[[trait_col, "x", "y"]].dropna()
    
    # Log the data structure before processing
    log.info("Trait data structure:")
    log.info("- Columns: %s", list(trait_df.columns))
    if hasattr(trait_df, 'npartitions'):
        log.info("- Partitions: %d", trait_df.npartitions)
    
    # Compute a sample to see the data structure
    sample_df = trait_df.head(10)
    log.info("Sample data (first 10 rows):")
    log.info("\n%s", sample_df.to_string())
    
    # Count total records
    total_records = len(trait_df)
    log.info("Total records for trait %s: %d", trait_col, total_records)
    
    if total_records == 0:
        log.warning("No data found for trait %s after dropping NaN values. Skipping.", trait_col)
        return

    # Step 2: Find spatial autocorrelation range for this trait
    log.info("Step 2: Looking up spatial autocorrelation range for trait %s", trait_col)
    log.info("Available traits in ranges DataFrame: %s", list(ranges['trait'].unique()))
    log.info("Range statistic being used: %s", cfg.train.cv_splits.range_stat)
    
    trait_range_rows = ranges[ranges["trait"] == trait_col]
    log.info("Matching rows for trait %s: %d", trait_col, len(trait_range_rows))
    
    if len(trait_range_rows) == 0:
        log.error("ERROR: Trait %s not found in spatial autocorrelation ranges!", trait_col)
        log.error("Available traits: %s", list(ranges['trait'].unique()))
        log.error("Ranges DataFrame shape: %s", ranges.shape)
        log.error("Ranges DataFrame columns: %s", list(ranges.columns))
        log.error("Sample of ranges DataFrame:")
        log.error("\n%s", ranges.head().to_string())
        raise ValueError(f"Trait {trait_col} not found in spatial autocorrelation ranges")
    
    trait_range = trait_range_rows[cfg.train.cv_splits.range_stat].values[0]
    log.info("Found spatial range for trait %s: %.2f meters", trait_col, trait_range)

    # Step 3: Adjust spatial range based on coordinate system and target resolution
    log.info("Step 3: Coordinate system processing")
    log.info("Current CRS: %s", cfg.crs)
    log.info("Target resolution: %.2f meters", cfg.target_resolution)
    log.info("Original trait range: %.2f meters", trait_range)
    
    if cfg.crs == "EPSG:4326":
        trait_range_deg = trait_range / 111320  # Convert meters to degrees
        log.info("Trait range in degrees: %.6f", trait_range_deg)
        if trait_range_deg <= cfg.target_resolution:
            log.warning(
                "Trait range of %.2f m (%.6f deg) is less than or equal to the existing map "
                "resolution of %.2f m. Using the map resolution for hexagon assignment...",
                trait_range, trait_range_deg, cfg.target_resolution,
            )
            trait_range = cfg.target_resolution * 111320

    if cfg.crs == "EPSG:6933":
        if trait_range <= cfg.target_resolution:
            log.warning(
                "Trait range of %.2f m is less than or equal to the existing map "
                "resolution of %.2f m. Using the map resolution for hexagon assignment...",
                trait_range, cfg.target_resolution,
            )
            trait_range = cfg.target_resolution

        log.info("Reprojecting coordinates from %s to WGS84 for hexagon assignment...", cfg.crs)
        log.info("Original coordinate columns: %s", list(trait_df.columns))
        
        # Convert coordinates to EPSG:4326 to get hexagon assignments
        # H3 hexagons require lat/lon coordinates
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
        
        # Rename columns: keep original coords as x_old/y_old, use lat/lon as x/y for H3
        trait_df = trait_df.rename(
            columns={"x": "x_old", "y": "y_old", "lat": "y", "lon": "x"}
        )
        log.info("After reprojection, columns: %s", list(trait_df.columns))

    # Step 4: Create hexagonal spatial clusters
    log.info("Step 4: Creating hexagonal spatial clusters")
    log.info("Final trait range for hexagon calculation: %.2f meters", trait_range)
    h3_res = acr_to_h3_res(trait_range)
    log.info("H3 hexagon resolution: %d", h3_res)
    log.info("Assigning hexagon IDs to data points...")
    
    trait_df = assign_hexagons(trait_df, h3_res, dask=True).reset_index(drop=True)
    log.info("After hexagon assignment, columns: %s", list(trait_df.columns))

    if cfg.crs == "EPSG:6933":
        # Revert back to the original coordinates for final output
        # We used lat/lon for H3 hexagons, but want to save original projected coords
        log.info("Reverting to original coordinate system for output...")
        trait_df = trait_df.drop(columns=["x", "y"]).rename(
            columns={"x_old": "x", "y_old": "y"}
        )
        log.info("Final columns after coordinate reversion: %s", list(trait_df.columns))

    if isinstance(trait_df, dd.DataFrame):
        log.info("Computing trait dask DataFrame (converting from lazy to concrete)...")
        trait_df = trait_df.compute()  # pyright: ignore[reportCallIssue]
        log.info("DataFrame computed. Shape: %s", trait_df.shape)

    # Step 5: Assign cross-validation folds to hexagons
    log.info("Step 5: Assigning CV folds to spatial clusters")
    log.info("Number of CV folds: %d", cfg.train.cv_splits.n_splits)
    log.info("Number of simulation iterations: %d", cfg.train.cv_splits.n_sims)
    log.info("Unique hexagons: %d", trait_df['hex_id'].nunique() if 'hex_id' in trait_df.columns else 0)
    
    trait_df = assign_folds(
        trait_df,
        cfg.train.cv_splits.n_splits,
        cfg.train.cv_splits.n_sims,
        trait_col,
    )
    
    log.info("Fold assignment completed. Fold distribution:")
    fold_counts = trait_df['fold'].value_counts().sort_index()
    for fold_id, count in fold_counts.items():
        log.info("  Fold %d: %d samples", fold_id, count)

    # Step 6: Save the splits
    log.info("Step 6: Saving CV splits to file")
    log.info("Columns before saving: %s", list(trait_df.columns))
    
    # We only need coordinates (x, y) and fold assignments for the splits file
    # The trait values themselves are not needed in the splits file
    # Each unique (x, y) location gets assigned to exactly one fold
    splits_df = trait_df[["x", "y", "fold"]].drop_duplicates(subset=["x", "y"]).reset_index(drop=True)
    
    log.info("Final splits data shape: %s", splits_df.shape)
    log.info("Unique locations: %d", len(splits_df))
    log.info("Sample of final splits data:")
    log.info("\n%s", splits_df.head().to_string())
    
    splits_df.to_parquet(splits_fn, compression="zstd")
    log.info("Splits saved to: %s", splits_fn)
    log.info("=" * 60)
    return None


if __name__ == "__main__":
    main()
