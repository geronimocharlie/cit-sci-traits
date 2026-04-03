# Spatial Cross-Validation

# 1 Spatial Autocorrelation Range Computation

### What is Spatial Autocorrelation?
Spatial autocorrelation measures how similar trait values are at different distances. The **autocorrelation range** is the distance at which observations become spatially independent (uncorrelated).

### Variogram-Based Range Calculation

#### Step 1: Data Preprocessing for Variogram Analysis
```python
# Extract sPlot data only (more reliable for variogram analysis)
y_ddf = dd.read_parquet(y_fn).query("source == 's'").drop(columns=["source"])

# For each trait, create clean dataset
trait_df = y_ddf[["x", "y", trait_col]].astype(np.float32).dropna()
# Result: ~15,000-25,000 sPlot observations per trait
```

**Why sPlot data only:**
- sPlot provides more controlled, standardized vegetation plot data
- GBIF citizen science data can be spatially biased or clustered
- Variogram requires relatively uniform spatial sampling
- sPlot data still sparce, but trait values more trustworthy

#### Step 2: Coordinate System Selection for Distance Calculations
```python
# The coordinate system choice affects distance calculations:

if cfg.crs == "EPSG:4326":  # Geographic coordinates (lat/lon)
    if cfg.target_resolution > 0.2:  # > 0.2 degrees (~22km)
        # Use Web Mercator (EPSG:3857) for large-scale analysis
        trait_df_wmerc = gpd.GeoDataFrame(trait_df, crs="EPSG:4326").to_crs("EPSG:3857")
    else:
        # Use UTM zones for accurate local distance calculations
        trait_df = add_utm(trait_df)  # Convert to appropriate UTM zones
        
elif cfg.crs == "EPSG:6933":  # Equal Area Cylindrical (already in meters)
    # Convert to positive coordinates (variogram needs positive values)
    trait_df = trait_df.assign(
        easting=trait_df.x + abs(trait_df.x.min()),
        northing=trait_df.y + abs(trait_df.y.min())
    )
```

#### Step 3: Variogram Computation Using Ordinary Kriging
```python
# Variogram parameters
vgram_kwargs = {
    "n_max": 18000,                    # Max points to use (computational limit)
    "variogram_model": "spherical",    # Theoretical model shape
    "nlags": 30,                       # Number of distance bins
    "anisotropy_scaling": 1,           # Assume isotropic spatial structure
    "anisotropy_angle": 0
}

# Fit variogram using PyKrige
ok_vgram = OrdinaryKriging(
    trait_df["easting"], 
    trait_df["northing"], 
    trait_df[trait_col], 
    **vgram_kwargs
)

# Extract range parameter from spherical model
autocorr_range = ok_vgram.variogram_model_parameters[1]
```

- foar each trait, for each point computing the parwise differences in distances and their variabilty (difference in actual trait value)
- where it plateus is the cutoff value -> for each trait, we have a range (in meters/km) and the variabilty

#### Step 4: Spherical Variogram Model
The spherical variogram model fits this equation to the data:
```python
# Spherical model: γ(h) = nugget + sill * [1.5*(h/range) - 0.5*(h/range)³]  if h ≤ range
#                        = nugget + sill                                        if h > range

# Where:
# γ(h) = semivariance at distance h
# nugget = measurement error + micro-scale variation  
# sill = total variance (nugget + spatial variance)
# range = distance where γ(h) reaches 95% of sill (effective independence)
```

#### Step 5: Spatial Chunking for Large Datasets
```python
# For global datasets, divide into spatial chunks to account for non-stationarity
if cfg.crs == "EPSG:6933" and syscfg.n_chunks > 1:
    # Create spatial grid (e.g., 16 chunks = 4x4 grid)
    x_bins = np.linspace(df.easting.min(), df.easting.max(), n_zones + 1)
    y_bins = np.linspace(df.northing.min(), df.northing.max(), n_zones // 2 + 1)
    
    # Assign each point to a spatial zone (from UTM zones)
    df["zone"] = [f"{x_zone}_{y_zone}" for x_zone, y_zone in zip(x_zones, y_zones)]
    
    # Compute separate variogram for each zone
    results = [calculate_variogram_pykrige(group, trait_col) for zone, group in df.groupby("zone")]
```

**Why Spatial Chunking is Necessary:**

1. **Non-Stationarity Problem**: Global ecological datasets violate the fundamental assumption of spatial stationarity
- Stationarity assumption: spatial correlation structure is the same everywhere
- Reality for global trait data:
    - Arctic tundra: short autocorr ranges due to harsh conditions
    - Tropical forests: longer ranges due to stable climate gradients  
    - Desert edges: abrupt changes in spatial structure
    - Mountain regions: elevation creates complex spatial patterns
   ```

2. **Computational Limitations**: Variogram calculation is O(n²) in complexity
- For 25,000 global sPlot points:
    - Single variogram: 25,000² = 625 million pairwise distance calculations
    - Chunked approach: 16 chunks × ~1,560² = ~39 million calculations total
    - Result: ~16x speedup with better spatial accuracy

3. **Improved Local Accuracy**: Each chunk captures regional spatial processes
   ```python
   # Example chunking for global dataset:
   chunks = {
       "0_0": "North America - Western",     # ~1,200 points
       "0_1": "North America - Eastern",     # ~1,800 points  
       "1_0": "Europe - Northern",           # ~2,100 points
       "1_1": "Europe - Southern",           # ~1,600 points
       "2_0": "Asia - Northern",             # ~900 points
       "2_1": "Asia - Tropical",             # ~1,400 points
       # ... 16 total chunks
   }
   # Each chunk gets its own variogram → more accurate local ranges
   ```

4. **Handling Geographic Barriers**: Chunking respects natural boundaries
-  Problems with global variogram:
    - Treats distance across Pacific Ocean same as across continent
    - Ignores mountain barriers, climate boundaries
    - Averages out distinct biogeographic regions
   
-  Chunking solution:
    - Each chunk represents coherent biogeographic region  
    - Respects natural spatial discontinuities
    - Accounts for regional differences in spatial processes
   ```

#### Step 6: Range Aggregation Across Spatial Chunks
```python
# Weight ranges by number of samples in each chunk
sample_sizes = np.array([n_samples for range_value, n_samples in results])
weights = sample_sizes / sample_sizes.sum()
ranges = np.array([range_value for range_value, n_samples in results])

# Calculate weighted statistics
final_range_stats = {
    "trait": trait_col,
    "mean": np.average(ranges, weights=weights),      # Primary range used for CV splits
    "std": np.sqrt(np.average((ranges - ranges.mean())**2, weights=weights)),
    "median": np.median(ranges),
    "q05": np.quantile(ranges, 0.05),
    "q95": np.quantile(ranges, 0.95),
    "n": sample_sizes.sum(),
    "n_chunks": len(ranges)
}
```

**Why Weighted Aggregation is Critical:**

1. **Unequal Sample Sizes Across Chunks**: Regional data availability varies dramatically
   ```python
   # Example chunk results for trait X1080_mean:
   chunk_results = [
       (850000, 2100),    # Europe: 850km range, 2,100 samples
       (1200000, 1800),   # North America: 1,200km range, 1,800 samples  
       (650000, 900),     # Northern Asia: 650km range, 900 samples
       (420000, 400),     # Australia: 420km range, 400 samples
       # ... 12 more chunks with varying sample sizes
   ]
   
   # Without weighting: simple mean = (850+1200+650+420)/4 = 780km
   # With weighting: accounts for data quality and reliability
   weights = [2100, 1800, 900, 400] / total_samples
   weighted_mean = (850×0.42 + 1200×0.36 + 650×0.18 + 420×0.08) = 894km
   ```

2. **Statistical Reliability**: More samples = more reliable variogram estimate

- Variogram reliability by sample size:
    - 200-500 samples: Basic variogram possible, high uncertainty
    - 500-1000 samples: Moderate reliability  
    - 1000-2000 samples: Good reliability for range estimation
    - 2000+ samples: Excellent reliability, stable parameters
   
- Weighting scheme gives more influence to well-sampled regions
   

3. **Geographic Representation**: Prevents bias toward data-sparse regions
- Problem without weighting:
    - 12 chunks with 100-300 samples each (sparse regions)
    - 4 chunks with 1500-2500 samples each (well-sampled regions)
- Simple average: 12×sparse + 4×dense = biased toward sparse regions
   
- Solution with weighting:
   - Sparse regions: low weight (uncertain estimates)
   - Dense regions: high weight (reliable estimates)  
   - Final result reflects data quality distribution


4. **Multiple Statistical Summaries**: Captures uncertainty and variability
   ```python
   aggregated_stats = {
       "mean": 946462.58,     # Weighted average - primary value for H3 sizing
       "std": 284731.42,      # Weighted standard deviation - uncertainty measure
       "median": 892341.45,   # Robust center - less sensitive to outliers
       "q05": 456789.23,      # 5th percentile - conservative estimate
       "q95": 1234567.89,     # 95th percentile - liberal estimate  
       "n": 18753,            # Total samples across all chunks
       "n_chunks": 16         # Number of spatial regions processed
   }
   
   # Range selection for CV splits:
   # - Primary: use "mean" (946 km) - balances all regional estimates
   # - Conservative: use "q05" (457 km) - ensures stronger independence
   # - Liberal: use "q95" (1,235 km) - allows more spatial correlation
   ```

### Coordinate System Impact on Variogram Analysis

#### 1. Geographic Coordinates (EPSG:4326) - Degrees
**Problems with direct lat/lon analysis:**

- Distance between points varies by latitude:
    - At equator: 1° ≈ 111 km
    - At 60°N: 1° longitude ≈ 55.6 km  
    - At 80°N: 1° longitude ≈ 19.3 km

- This creates distorted variogram calculations


**Solution - UTM Conversion:**
```python
def add_utm(df):
    """Convert lat/lon to UTM coordinates for accurate distance calculations"""
    for lat, lon in zip(df.y, df.x):
        easting, northing, zone, letter = utm.from_latlon(lat, lon)
        # Each UTM zone provides accurate metric coordinates within ~6° longitude bands
    return df  # Now with accurate meter-based coordinates
```

#### 2. Equal Area Cylindrical (EPSG:6933) - Already in Meters
**Advantages:**
- Coordinates already in meters globally
- Equal-area projection preserves relative spatial relationships
- No need for coordinate transformation

**Implementation for variogram:**
```python
# EPSG:6933 coordinates can be negative, but variogram needs positive values
trait_df = trait_df.assign(
    easting=trait_df.x + abs(trait_df.x.min()),  # Shift to positive
    northing=trait_df.y + abs(trait_df.y.min())
)

# Now distances are calculated correctly in meters:
distance_km = np.sqrt((easting1 - easting2)**2 + (northing1 - northing2)**2) / 1000
```

### What Would "Equal-Distant" Coordinate System Mean?

An **equal-distant (equidistant)** coordinate system preserves distances from a central point, but not between all points. For variogram analysis, this would mean:

#### Theoretical Implementation:
```python
def implement_equidistant_variogram(central_point_lat, central_point_lon):
    """
    Implement variogram in equidistant projection centered on study area
    """
    # Define equidistant projection centered on study area
    central_proj = f"+proj=aeqd +lat_0={central_point_lat} +lon_0={central_point_lon} +datum=WGS84"
    
    # Convert all points to this projection
    trait_gdf = gpd.GeoDataFrame(
        trait_df, 
        geometry=gpd.points_from_xy(trait_df.x, trait_df.y),
        crs="EPSG:4326"
    ).to_crs(central_proj)
    
    # Extract coordinates (now distances are accurate from center point)
    trait_df["easting"] = trait_gdf.geometry.x  # meters from center
    trait_df["northing"] = trait_gdf.geometry.y  # meters from center
    
    # Compute variogram using these "equidistant-from-center" coordinates
    return calculate_variogram_pykrige(trait_df, trait_col)
```

#### Practical Implications:
- **Accurate near center**: Distances accurate close to the central point
- **Increasing distortion**: Distances become less accurate farther from center  
- **Global datasets**: Not ideal for worldwide trait data spanning continents
- **Regional analysis**: Could be useful for country or continent-scale studies

#### Why Current Approach is Better:
The current implementation using **UTM zones** (for EPSG:4326) or **Equal Area Cylindrical** (EPSG:6933) is more appropriate because:
- UTM provides accurate metric distances within each 6° longitude zone
- Equal Area Cylindrical preserves spatial relationships globally
- Both avoid the center-point bias of equidistant projections
- Better suited for global ecological datasets

#### Step 7: Final Output Data Structure

The spatial autocorrelation calculation produces a single comprehensive file containing range statistics for all traits:

**Output File Structure:**
```python
# File: reference/spatial_autocorr_Shrub_Tree_Grass_55km_mean.parquet
# Structure: pandas DataFrame with one row per trait

spatial_autocorr_results = pd.DataFrame({
    'trait': ['X1080_mean', 'X138_mean', 'X13_mean', ...],     # 37 trait identifiers
    'mean': [946462.58, 823451.23, 1124567.89, ...],          # Weighted mean ranges (meters)
    'std': [234567.12, 187432.45, 298765.43, ...],            # Weighted standard deviations
    'median': [892341.45, 798234.56, 1087654.32, ...],        # Median ranges (robust estimator)
    'q05': [456789.23, 398765.43, 567834.21, ...],           # 5th percentile (conservative)
    'q95': [1234567.89, 1156789.43, 1687543.21, ...],        # 95th percentile (liberal)
    'n': [18753, 17892, 16543, ...],                         # Total samples per trait
    'n_chunks': [16, 15, 14, ...]                            # Chunks with valid variograms
})

# Shape: (37 rows × 8 columns)
# Size: ~3-5 KB (small reference file)
```

**Key Properties of Final Dataset:**

1. **Trait Coverage**: One row per trait in the original dataset
   ```python
   # All 37 traits from Y.parquet get spatial autocorr statistics:
   traits_covered = [
       'X1080_mean',  # Root length per dry mass  
       'X138_mean',   # Seed number per reproduction unit
       'X13_mean',    # Leaf carbon content
       'X144_mean',   # Leaf length
       # ... 33 more trait columns
   ]
   # Missing traits = traits with insufficient spatial data for variogram
   ```

2. **Range Statistics Interpretation**:
   ```python
   # For each trait, multiple range estimates provide flexibility:
   
   trait_example = {
       'trait': 'X1080_mean',
       'mean': 946462.58,      # ← PRIMARY: Used for CV split hexagon sizing
       'std': 234567.12,       # Uncertainty: ±235 km around mean estimate  
       'median': 892341.45,    # Robust: Less sensitive to extreme chunks
       'q05': 456789.23,       # Conservative: Ensures strong independence
       'q95': 1234567.89,      # Liberal: Allows more spatial correlation
       'n': 18753,             # Data quality: Higher n = more reliable
       'n_chunks': 16          # Geographic coverage: More chunks = global representation
   }
   
   # Range selection impact on CV splits:
   # mean (946km) → H3 resolution 0 (1,107km hexagons) → ~102 global hexagons
   # q05 (457km)  → H3 resolution 1 (418km hexagons)  → ~847 global hexagons  
   # q95 (1,235km)→ H3 resolution 0 (1,107km hexagons) → ~102 global hexagons
   ```

3. **Quality Indicators**: Sample size and chunk coverage reveal data reliability
   ```python
   # High-quality traits (reliable range estimates):
   high_quality = {
       'n': 15000+,           # Many sPlot observations available
       'n_chunks': 12+,       # Good geographic coverage  
       'std/mean': <0.3       # Consistent ranges across regions
   }
   
   # Lower-quality traits (less reliable):  
   lower_quality = {
       'n': 3000-8000,       # Fewer observations
       'n_chunks': 6-10,     # Limited geographic coverage
       'std/mean': >0.5      # High variability between regions
   }
   ```

4. **File Usage in Pipeline**: This file becomes the lookup table for CV splitting
   ```python
   # In skcv_splits.py:
   ranges = pd.read_parquet("reference/spatial_autocorr_Shrub_Tree_Grass_55km_mean.parquet")
   
   # For each trait during CV split generation:
   trait_range = ranges[ranges["trait"] == "X1080_mean"]["mean"].values[0]
   # Returns: 946462.58 meters
   
   # This drives the entire spatial clustering process:
   # 946km → H3 resolution 0 → ~1,107km hexagons → spatial independence
   ```

**Final Range Values and Usage:**
```python
# The complete spatial autocorrelation results enable:

1. **Trait-Specific Spatial Clustering**: Each trait gets appropriate hexagon size
2. **Quality Assessment**: Identify traits with reliable vs. uncertain spatial structure  
3. **Method Validation**: Compare range estimates across different statistical approaches
4. **Future Flexibility**: Switch between conservative/liberal spatial independence criteria

# Example usage in CV splits:
trait_range = 946462.58  # meters from autocorrelation analysis
h3_resolution = acr_to_h3_res(trait_range)  # Converts to H3 resolution 0  
hexagon_edge_length = 1107000  # meters (H3 res 0 specification)
# Result: Points >946km apart are in different hexagons → spatially independent CV folds
```

# Splits Process - Technical Documentation

## Overview

This document explains how spatial cross-validation splits are created from original trait data for the Shrub Tree Grass 55km dataset. The process creates spatial fold assignments that minimize spatial autocorrelation for robust machine learning model validation.

## 1. Data Input Structure

### Original Data Format
- **File**: `data/features/Shrub_Tree_Grass/55km/Y.parquet`
- **Structure**: Dask DataFrame with 100 partitions
- **Columns**: 
  - `x`, `y`: Spatial coordinates in EPSG:6933 (Equal Area Cylindrical)
  - `X1080_mean`, `X138_mean`, etc.: 37 trait columns (standardized trait values)
  - `source`: Data source indicator (`'g'` = GBIF)

```python
# Sample data structure
trait_data = {
    'x': [-1.733996e+07, -1.733996e+07, ...],  # Projected coordinates
    'y': [6.980914e+06, 6.925289e+06, ...], 
    'X1080_mean': [0.622779, 0.549480, ...],    # Trait values (standardized)
    'source': ['g', 'g', ...]
}
```


## 2 Step-by-Step Splits Creation Process

### Step 1: Data Extraction and Filtering
```python
# Extract trait data for specific trait + coordinates
trait_df = traits_df[[trait_col, "x", "y"]].dropna()
# Result: DataFrame with ~45,000 records per trait
```

**What happens:**
- Selects one trait column plus coordinates (`x`, `y`)
- Removes rows with missing trait values (`dropna()`)
- Typical result: ~45,000 valid observations per trait

### Step 2: Spatial Autocorrelation Range Lookup
```python
# Load pre-computed spatial ranges
ranges = pd.read_parquet("spatial_autocorr.parquet", columns=["trait", "mean"])
trait_range = ranges[ranges["trait"] == trait_col]["mean"].values[0]
# Example: X1080_mean → 946,462.58 meters
```

**What happens:**
- Looks up pre-computed spatial autocorrelation range for the trait
- This range defines the distance at which observations become spatially independent
- Used to determine appropriate hexagon size for spatial clustering

### Step 3: Coordinate System Processing
```python
# Convert EPSG:6933 → EPSG:4326 for H3 hexagon assignment
trait_df = trait_df.map_partitions(reproject_xy_to_geo, from_crs="EPSG:6933", meta=meta)
trait_df = trait_df.rename(columns={"x": "x_old", "y": "y_old", "lat": "y", "lon": "x"})
```

**What happens:**
- **EPSG:6933**: World Equidistant Cylindrical projection (Equal Area Cylindrical)
  - Coordinates in meters (e.g., x: -17,339,960 m, y: 6,980,914 m)
  - Preserves areas accurately for spatial analysis
  - Used for distance calculations and spatial autocorrelation analysis
- **EPSG:4326**: World Geodetic System 1984 (WGS84) - Geographic coordinate system
  - Coordinates in decimal degrees (e.g., lon: -155.2°, lat: 62.8°)
  - Standard latitude/longitude system used by GPS
  - Required by H3 hexagonal indexing system
- **Why reprojection is needed**: H3 library only accepts geographic coordinates (lat/lon in degrees)
- Keeps original projected coordinates as `x_old`, `y_old` for final output


## BACKGROUND Coordinate Systems Explained

### EPSG:6933 - World Equidistant Cylindrical (Equal Area Cylindrical)
```python
# Example coordinates in EPSG:6933:
x = -17339960.0  # meters (negative = west of prime meridian)
y = 6980914.0    # meters (positive = north of equator)
```

**Properties:**
- **Units**: Meters from a central reference point
- **Type**: Projected coordinate system (3D Earth → 2D plane)
- **Area Preservation**: Equal-area projection - preserves relative sizes of regions
- **Distance Calculations**: Accurate for computing distances and spatial autocorrelation
- **Coverage**: Global, but with increasing distortion away from standard parallels

**Why Used for Analysis:**
- Spatial autocorrelation ranges are measured in meters (e.g., 946,462.58 m)
- Distance-based clustering algorithms need metric coordinates
- Area calculations for hexagons are more accurate

### EPSG:4326 - World Geodetic System 1984 (WGS84)
```python
# Same location in EPSG:4326:
longitude = -155.2°  # degrees west 
latitude = 62.8°     # degrees north
```

**Properties:**
- **Units**: Decimal degrees (latitude: -90° to +90°, longitude: -180° to +180°)
- **Type**: Geographic coordinate system (spherical coordinates on Earth ellipsoid)
- **Standard**: Global standard used by GPS, Google Maps, etc.
- **No Projection**: Represents actual curved Earth surface, not flattened
- **Universal**: Every location on Earth has unique lat/lon coordinates

**Why Required for H3:**
- H3 library was designed around geographic coordinates
- Hexagon boundaries are defined on the Earth's curved surface
- More intuitive for global applications (latitude/longitude familiar to users)

### Coordinate Transformation Process
```python
def reproject_xy_to_geo(df, from_crs="EPSG:6933", to_crs="EPSG:4326"):
    """Convert projected coordinates to geographic coordinates"""
    # Input: df with 'x', 'y' columns in meters
    # Output: df with 'lon', 'lat' columns in degrees
    
    # Example transformation:
    # EPSG:6933: x=-17,339,960m, y=6,980,914m 
    # ↓ (mathematical transformation using geodetic formulas)
    # EPSG:4326: lon=-155.2°, lat=62.8°
```

### Step 4: Hexagonal Spatial Clustering
```python
# Convert range to H3 resolution
h3_res = acr_to_h3_res(trait_range)  # Usually results in resolution 0
# Assign hexagon IDs
trait_df = assign_hexagons(trait_df, h3_res, dask=True)
```

**What happens:**

#### H3 Resolution Selection
```python
# Example: trait_range = 946,462.58 meters (~940km)
# H3 resolutions and their approximate edge lengths:
# Resolution 0: ~1,107 km edge length (largest hexagons)
# Resolution 1: ~418 km edge length  
# Resolution 2: ~158 km edge length
# Since 940km > 418km, algorithm selects resolution 0
```

#### Hexagon Assignment Process
```python
# For each data point (lat, lon):
lat, lon = 62.8, -155.2  # Example coordinates in Alaska
hex_id = h3.geo_to_h3(lat, lon, h3_res)  
# Returns: '8001fffffffffff' (unique 15-character hexagon identifier)
```

#### H3 Hexagon Properties
- **Shape**: Regular hexagons that tile the entire Earth surface
- **Hierarchy**: Each hexagon contains exactly 7 child hexagons at the next resolution
- **Unique IDs**: 15-character identifiers (e.g., '8001fffffffffff', '8003fffffffffff')
- **Global Coverage**: 122 base hexagons at resolution 0 cover the entire planet
- **Area Consistency**: All hexagons at same resolution have approximately equal area

#### Spatial Clustering Results
- **Input**: ~45,000 data points with (lat, lon) coordinates
- **Output**: Each point assigned to one of ~101-102 unique H3 hexagons
- **Clustering**: Points within ~940km of each other likely end up in the same hexagon
- **Coverage**: Global dataset spans multiple continents, hence many hexagons needed

#### Why Hexagons Instead of Squares?
- **Better Neighbors**: Each hexagon has 6 equidistant neighbors (vs 4 + 4 diagonal for squares)
- **Less Distortion**: More uniform shape when projected on curved Earth surface
- **Better Clustering**: Circular regions map more naturally to hexagons than squares
- **Industry Standard**: Used by Uber, Meta, and other companies for geospatial analysis


### Step 5: Cross-Validation Fold Assignment
```python
# Optimize fold assignments using Kolmogorov-Smirnov similarity
trait_df = assign_folds(
    trait_df, 
    n_splits=5,           # 5-fold CV
    n_sims=200,           # 200 optimization iterations
    trait_col             # Trait column for similarity optimization
)
```

**What happens:**
- Randomly assigns hexagons to 5 CV folds (200 different random assignments)
- For each assignment, calculates Kolmogorov-Smirnov test p-values between all fold pairs
- Selects assignment with highest mean p-value (most similar trait distributions)
- Each data point gets assigned to exactly one fold (0, 1, 2, 3, or 4)

### Step 6: Final Data Structure and Saving
```python
# Revert to original coordinate system and save
trait_df = trait_df.drop(columns=["x", "y"]).rename(columns={"x_old": "x", "y_old": "y"})
splits_df = trait_df[["x", "y", "fold"]].drop_duplicates(subset=["x", "y"])
splits_df.to_parquet(f"data/features/Shrub_Tree_Grass/55km/skcv_splits/{trait_col}.parquet")
```

## 3. Output Data Structure

### Individual Split Files
- **Location**: `data/features/Shrub_Tree_Grass/55km/skcv_splits/`
- **Files**: One per trait (e.g., `X1080_mean.parquet`, `X138_mean.parquet`)
- **Structure**: 

```python
split_file_structure = {
    'x': float64,          # Original EPSG:6933 x-coordinates  
    'y': float64,          # Original EPSG:6933 y-coordinates
    'fold': int32          # Fold assignment (0, 1, 2, 3, or 4)
}
# Shape: (~45,000 rows, 3 columns) per trait
```

### Fold Distribution
Each split file contains approximately:
- **Fold 0**: ~9,000 locations
- **Fold 1**: ~9,000 locations  
- **Fold 2**: ~9,000 locations
- **Fold 3**: ~9,000 locations
- **Fold 4**: ~9,000 locations
