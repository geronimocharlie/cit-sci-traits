# Complete Trait Mapping Pipeline: From Raw Data to Spatial Cross-Validation

## Overview

This document provides a comprehensive explanation of the global plant trait mapping pipeline, from raw data processing through spatial cross-validation design. The pipeline creates global maps of community-weighted mean (CWM) plant traits at multiple spatial resolutions (1-222 km) using machine learning models trained on ecological survey data (sPlot), citizen science observations (GBIF), and Earth observation data.

**Key Innovation**: The pipeline addresses spatial autocorrelation in ecological data through variogram-based spatial cross-validation, ensuring robust model evaluation for global trait mapping applications.

---

## 1. Pipeline Architecture

The pipeline follows a structured data flow:

```
Raw Data → Interim Processing → Feature Engineering → Model Training → Spatial Cross-Validation → Final Products
```

### 1.1 Data Sources

**Biological Data:** (Targets)
- **sPlot**: Vegetation plot surveys with species abundance/cover data (~95,000 plots globally)
- **GBIF**: Citizen science species occurrence records (~50 million records)
- **TRY**: Plant trait database with species-level measurements (~31 functional traits)

**Environmental Data:** (Predictors)
- **MODIS**: Satellite reflectance (500m-1km native resolution)
- **WorldClim**: Bioclimatic variables (~1km native resolution) 
- **SoilGrids**: Soil properties (250m native resolution)
- **VODCA**: Vegetation optical depth (~25km native resolution)
- **ETH Global Canopy Height**: Forest structure (~1km native resolution)

### 1.2 Target Resolutions

Models are trained independently at multiple spatial resolutions:
- **1km**: Fine-scale local patterns
- **22km**: Regional landscape patterns  
- **55km**: Sub-continental patterns
- **111km**: Continental patterns
- **222km**: Global biogeographic patterns

**Why multiple resolutions?** This avoids scale-transfer bias from naive upscaling and captures scale-dependent ecological processes.

### 1.3 Temporal Handling Strategy

**Important design decision**: All data are treated as **temporally static** for this global analysis.

**Biological Data Temporal Treatment:**
- **GBIF**: Spans 1700s to present, but records contribute equally regardless of sampling year
- **sPlot**: Surveys from 1920s to present, treated as snapshots (not time series)
- **TRY**: Decades of trait measurements, but traits assumed static for species

**Environmental Data Temporal Aggregation:**
- **MODIS**: Daily/16-day data aggregated to long-term climatological means per grid cell
- **WorldClim**: 1950-2000 climate normals (already temporally averaged)
- **VODCA**: 1987-2016 multi-year mean vegetation optical depth
- **SoilGrids**: Essentially static soil properties

**Key Implication**: The pipeline produces **static trait maps** rather than time-varying predictions. This approach is necessary because:
1. Trait measurements are too temporally sparse for time-series modeling
2. Most vegetation surveys are single snapshots, not repeated monitoring
3. Focus is on spatial patterns rather than temporal dynamics

**Future Potential**: The authors note that incorporating temporal dynamics (e.g., MODIS seasonality) is promising for future work, but current trait data limitations prevent this.

---

## 2. Data Processing Pipeline

### 2.1 Raw Data Harmonization

#### Earth Observation Data Processing
```python
# Example from harmonize_eo_data.py
def harmonize_eo_datasets(target_resolution, target_crs="EPSG:6933"):
    """
    Harmonize multiple EO datasets to common grid
    """
    # Reproject all datasets to Equal Area Cylindrical (EPSG:6933)
    for dataset in ["modis", "worldclim", "soilgrids", "vodca", "canopy_height"]:
        raster = load_raster(dataset)
        
        # Reproject to target CRS
        raster_reprojected = raster.rio.reproject(target_crs)
        
        # Resample to target resolution using appropriate method
        if dataset in ["modis", "canopy_height"]:
            # Use bilinear for continuous variables
            raster_resampled = raster_reprojected.rio.reproject_match(
                target_grid, resampling=Resampling.bilinear
            )
        else:
            # Use nearest neighbor for categorical or discrete data
            raster_resampled = raster_reprojected.rio.reproject_match(
                target_grid, resampling=Resampling.nearest
            )
```

**Key Processing Steps:**
1. **Reprojection**: Convert all datasets to EPSG:6933 (World Equidistant Cylindrical)
   - Preserves areas globally for accurate spatial analysis
   - Enables metric distance calculations essential for spatial autocorrelation
2. **Resampling**: Aggregate/disaggregate to target resolution
   - Conservative resampling (mean, median) for continuous variables
   - Nearest neighbor for categorical data
3. **Masking**: Apply study area boundaries using ESA WorldCover land use data

#### Species Occurrence Processing
```python
# Example from match_gbif_pfts.py and build_gbif_maps.py
def process_gbif_occurrences(pft_filter="Shrub_Tree_Grass"):
    """
    Process GBIF occurrences with trait matching
    """
    # Load and filter GBIF data by Plant Functional Type
    gbif_data = dd.read_parquet("data/raw/all_tracheophyta_non-cult_2024-04-10.parquet")
    gbif_filtered = gbif_data.query(f"pft == '{pft_filter}'")
    
    # Spatial filtering and reprojection
    if cfg.crs != "EPSG:4326":
        # Convert lat/lon to projected coordinates for spatial binning
        gbif_projected = reproject_coordinates(
            gbif_filtered, from_crs="EPSG:4326", to_crs=cfg.crs
        )
    
    # Spatial binning to target resolution
    gbif_gridded = assign_grid_cells(
        gbif_projected, resolution=cfg.target_resolution
    )
    
    # Quality control: subsample dense cells to avoid bias
    gbif_balanced = gbif_gridded.groupby("grid_cell").apply(
        lambda x: x.sample(min(len(x), cfg.max_count_per_cell))
    )
    
    return gbif_balanced
```

### 2.2 Trait Integration and Community-Weighted Means

#### sPlot: True Community-Weighted Means
```python
# Example from build_splot_maps.py
def compute_splot_cwms(plot_data, trait_data):
    """
    Compute true community-weighted means from plot surveys
    """
    cwms_per_plot = []
    
    for plot_id, plot in plot_data.groupby("PlotObservationID"):
        # Get species abundances in plot
        species_abundances = plot[["Species", "Cover_perc"]].set_index("Species")
        
        # Match species to traits
        plot_traits = species_abundances.join(trait_data, how="inner")
        
        # Compute CWM: weighted by relative abundance/cover
        for trait in trait_columns:
            if len(plot_traits[trait].dropna()) > 0:
                cwm_value = np.average(
                    plot_traits[trait].dropna(),
                    weights=plot_traits.loc[plot_traits[trait].notna(), "Cover_perc"]
                )
                cwms_per_plot.append({
                    "PlotID": plot_id,
                    "trait": trait, 
                    "CWM": cwm_value,
                    "x": plot["x"].iloc[0],
                    "y": plot["y"].iloc[0]
                })
    
    return pd.DataFrame(cwms_per_plot)
```

#### GBIF: Frequency-Weighted Means  
```python
# Example from build_gbif_maps.py
def compute_gbif_frequency_means(occurrence_data, trait_data):
    """
    Compute frequency-weighted means from occurrence records
    """
    # Join occurrences with trait data
    occurrences_with_traits = occurrence_data.merge(
        trait_data, left_on="species", right_index=True, how="inner"
    )
    
    # Aggregate by grid cell
    grid_aggregated = occurrences_with_traits.groupby("grid_cell").agg({
        trait: ["mean", "count", "std"] for trait in trait_columns
    })
    
    return grid_aggregated
```

**Key Difference:**
- **sPlot**: Weighted by actual species abundance/cover in vegetation surveys (true CWMs)
- **GBIF**: Weighted by observation frequency - biased by sampling effort (frequency-weighted means)

**Detailed Spatial Aggregation Process:**

**For each grid cell (1-222 km):**

1. **Species Occurrence Binning**: Point-level observations are assigned to grid cells
   - **sPlot**: Species relative abundance/cover within vegetation plots
   - **GBIF**: Species occurrence counts (presence-only, no abundance data)

2. **Trait Matching**: Species in each grid cell are linked to their traits from TRY database

3. **Community-Weighted Mean Calculation**: 
   $$\text{CWM}_{\text{cell}} = \sum_{i} p_i \times t_i$$
   where $p_i$ = relative weight of species $i$ and $t_i$ = trait value for species $i$

4. **Environmental Predictor Aggregation**: All environmental layers resampled to same grid
   - **MODIS reflectance** (500m-1km native) → averaged over pixels within cell
   - **SoilGrids** (250m native) → mean/median aggregation within cell  
   - **VODCA** (25km native) → downscaled/upscaled to align with target grid
   - **WorldClim** (~1km native) → aggregated to coarser grids
   - **Canopy Height** (~1km native) → mean height within cell

**Critical Design Choice**: Instead of producing a single high-resolution product and upscaling, the study trains **independent models at each resolution**. This avoids scale-transfer bias and smoothing artifacts from naive upscaling.

---

## 3. Spatial Autocorrelation Analysis

### 3.1 The Spatial Autocorrelation Problem

**What is spatial autocorrelation?** It's the tendency for nearby locations to have similar trait values. In simple terms: if you know the leaf size at one forest location, you can make a pretty good guess about leaf sizes at nearby locations, but not at distant ones.

Ecological data violate the **Independent and Identically Distributed (IID)** assumption of standard machine learning due to **spatial autocorrelation**. Mathematically, this means:

$$\text{Cov}(Y_i, Y_j) \neq 0 \text{ when } |location_i - location_j| < \text{range}$$

**Why this matters for machine learning:**

1. **Inflated performance metrics**: If training and test points are close together, the model appears to work better than it actually does
2. **Poor generalization**: Models fail when predicting in new geographic regions they haven't seen before  
3. **Overconfident uncertainty estimates**: We underestimate how uncertain our predictions really are

**Simple analogy**: It's like testing a student's math skills by giving them practice problems that are almost identical to the homework - they might score well on the test, but struggle with truly new problems.

### 3.2 Variogram-Based Range Estimation

The pipeline uses **variogram analysis** to quantify spatial autocorrelation ranges for each trait:

```python
# From calc_spatial_autocorr.py
def calculate_variogram_pykrige(group, trait_col, **kwargs):
    """
    Fit spherical variogram model to estimate spatial range
    """
    if len(group) < 200:  # Minimum sample size for reliable variogram
        return 0, 0
    
    # Subsample if too many points (computational efficiency)
    if len(group) > 20000:
        group = group.sample(20000)
    
    # Fit variogram using Ordinary Kriging
    ok_vgram = OrdinaryKriging(
        group["easting"],      # x-coordinates in meters
        group["northing"],     # y-coordinates in meters  
        group[trait_col],      # trait values
        variogram_model="spherical",  # Theoretical model
        nlags=min(50, max(10, len(group) // 20)),  # Distance bins
        **kwargs
    )
    
    # Extract range parameter (distance where correlation drops to ~5%)
    autocorr_range = ok_vgram.variogram_model_parameters[1]  # meters
    n_samples = len(group)
    
    return autocorr_range, n_samples
```

#### Spherical Variogram Model

**What is a variogram?** Think of it as a "similarity decay function" - it measures how quickly two locations become different as you increase the distance between them.

The spherical model describes how spatial similarity decreases with distance:

$$\gamma(h) = \begin{cases}
\text{nugget} + \text{sill} \times \left[1.5 \times \frac{h}{\text{range}} - 0.5 \times \left(\frac{h}{\text{range}}\right)^3\right] & \text{if } h \leq \text{range} \\
\text{nugget} + \text{sill} & \text{if } h > \text{range}
\end{cases}$$

**Parameter meanings in simple terms:**
- **$h$** = distance between points (meters)
- **$\gamma(h)$** = semivariance (how different two points are at distance $h$)
- **nugget** = baseline "noise" - even identical locations have some variation
- **sill** = maximum difference you'd expect between any two random points
- **range** = the "magic distance" where locations stop being similar to each other

**Physical Interpretation:**
- **Short distances** ($h <$ range): Points are spatially correlated (similar to each other)
- **Long distances** ($h >$ range): Points are spatially independent (no more similar than random)
- **Range parameter**: The critical distance we use to design spatial cross-validation - this tells us how far apart training and test data need to be

### 3.3 Coordinate System Requirements

Accurate variogram computation requires **metric coordinates** (distances in meters):

```python
# From calc_spatial_autocorr.py - Coordinate system handling
def handle_coordinate_systems(trait_df, cfg):
    """
    Convert coordinates to appropriate system for distance calculations
    """
    if cfg.crs == "EPSG:4326":  # Geographic coordinates (degrees)
        if cfg.target_resolution > 0.2:  # Coarse resolution (>22km)
            # Use Web Mercator projection for global analysis
            trait_df_projected = convert_to_web_mercator(trait_df)
            
        else:  # Fine resolution
            # Use UTM zones for accurate local distance calculations  
            trait_df_utm = add_utm_coordinates(trait_df)
            # Each UTM zone provides metric coordinates within ~6° longitude
            
    elif cfg.crs == "EPSG:6933":  # Equal Area Cylindrical (already metric)
        # Convert to positive coordinates (variogram algorithm requirement)
        trait_df["easting"] = trait_df.x + abs(trait_df.x.min())
        trait_df["northing"] = trait_df.y + abs(trait_df.y.min())
        
    return trait_df
```

**Why EPSG:6933 for Global Analysis?**

**Simple explanation**: Think of projections like different ways to flatten a globe onto a map. Each method distorts something - either shapes, areas, or distances.

- **Equal-area projection**: A 1km² forest in Canada appears the same size as a 1km² forest in Brazil (preserves relative sizes globally)
- **Metric coordinates**: Distances are in meters everywhere, so we can directly calculate "this forest is 847,392 meters from that one"  
- **Global coverage**: One coordinate system covers the entire world (no awkward boundaries between different zones)

**Why this matters**: When we compute variograms, we need accurate distances in meters. If we used latitude/longitude, 1 degree near the equator would be much longer than 1 degree near the poles, messing up our spatial autocorrelation calculations.

### 3.4 Spatial Chunking for Non-Stationarity

**The Core Problem**: Global datasets exhibit **spatial non-stationarity** - spatial autocorrelation structure varies dramatically across different regions of Earth.

**Simple Explanation**: Think of it like climate zones. The distance over which forest properties stay similar varies by continent:
- **Boreal forests** (Canada, Siberia): Very homogeneous over vast distances (1,400+ km)
- **European landscapes**: Fragmented by agriculture and urbanization (850 km ranges)
- **Tropical regions**: High environmental gradients create shorter autocorrelation ranges (920 km)
- **Arid Australia**: Sharp environmental transitions limit similarity ranges (650 km)

#### How Spatial Chunking Works in the Pipeline

**Step 1: Geographic Subdivision**
```python
# From calc_spatial_autocorr.py - Actual implementation
def _assign_zones(df: pd.DataFrame, n_zones: int) -> pd.DataFrame:
    """
    Divide global data into rectangular spatial zones
    """
    # Create grid boundaries 
    x_bins = np.linspace(df.easting.min(), df.easting.max(), n_zones + 1)
    y_bins = np.linspace(df.northing.min(), df.northing.max(), n_zones // 2 + 1)
    
    # Assign each data point to a zone
    x_zones = np.digitize(df.easting, x_bins) - 1  # Which x-bin?
    y_zones = np.digitize(df.northing, y_bins) - 1  # Which y-bin?
    
    # Create unique zone identifiers like "0_1", "2_3", etc.
    df["zone"] = [f"{x}_{y}" for x, y in zip(x_zones, y_zones)]
    return df

# Example: For n_chunks=4, creates 4 rectangular zones globally:
# Zone "0_0": Northwestern quadrant (e.g., North America + northern Eurasia)
# Zone "1_0": Northeastern quadrant (e.g., eastern Asia + Alaska)  
# Zone "0_1": Southwestern quadrant (e.g., South America + western Africa)
# Zone "1_1": Southeastern quadrant (e.g., southern Africa + Australia)
```

**Step 2: Compute Separate Variograms per Zone**
```python
# Each geographic zone gets its own variogram analysis
if syscfg.n_chunks > 1:  # n_chunks comes from params.yaml
    trait_df_grouped = trait_df.pipe(_assign_zones, n_zones=syscfg.n_chunks).groupby("zone")
    
    # Compute variogram for each zone separately
    results = [
        calculate_variogram_pykrige(group, trait_col, **vgram_kwargs)
        for _, group in trait_df_grouped  # Each 'group' is one geographic zone
    ]
else:
    # Single global variogram (no chunking)
    results = [calculate_variogram_pykrige(trait_df, trait_col, **vgram_kwargs)]
```

**Why This Matters**: Each zone produces its own range estimate reflecting regional ecological patterns.

**Example Regional Variation Results**:
```python
# Real output from running spatial chunking (hypothetical trait):
zone_results = {
    "0_0": (1200000, 8453),    # 1,200 km range, 8,453 data points (North America/Eurasia)
    "1_0": (1400000, 3287),    # 1,400 km range, 3,287 data points (East Asia/Alaska)  
    "0_1": (920000, 6542),     # 920 km range, 6,542 data points (South America/West Africa)
    "1_1": (650000, 4891),     # 650 km range, 4,891 data points (Australia/Southern Africa)
}
```

#### Configuration and Workflow Integration

**Configuration Control** (from `params.yaml`):
```yaml
calc_spatial_autocorr:
  n_workers: 1
  n_chunks: 4        # ← Controls spatial subdivision 
  gpu_ids: []
```

**Resolution-Dependent Chunking Strategy**:
- **Fine resolutions** (1-22 km): `n_chunks: 1` (single global variogram)
  - Small-scale patterns are relatively consistent globally
  - Sufficient local data density for stable variograms
- **Coarse resolutions** (55-222 km): `n_chunks: 4` (regional variograms)  
  - Continental-scale patterns vary dramatically between regions
  - Need regional adaptation for cross-validation design

#### Workflow Position: Where Chunking Fits

**Complete Spatial Autocorrelation Workflow**:
```
1. Load trait data (sPlot vegetation surveys) → trait_df
2. Handle coordinate system (EPSG:6933 → easting/northing)
3. **SPATIAL CHUNKING** (if n_chunks > 1):
   - Divide data into geographic zones
   - Group data by zone
4. **VARIOGRAM COMPUTATION** (per zone):
   - Fit spherical model to each zone's data
   - Extract range parameter for each zone
5. **RANGE AGGREGATION** (next section):
   - Combine zone-specific ranges using weighted averaging
   - Produce final autocorrelation range for cross-validation
6. **SPATIAL CV DESIGN**:
   - Use aggregated range → H3 hexagon sizing → CV folds
```

**Critical Design Decision**: Chunking happens **before** variogram computation, not after. This ensures each region's unique spatial structure is properly captured rather than averaged away in a global analysis.

### 3.5 Weighted Range Aggregation

**Connecting Chunking to Cross-Validation**: After computing separate variograms for each geographic zone, regional range estimates must be combined into a single value for H3 hexagon sizing.

**The Aggregation Problem**: Different regions have different data densities and autocorrelation patterns. How do we fairly combine them?

**Solution**: **Sample-size weighted averaging** - regions with more data points get higher influence in the final range estimate.

```python
# From calc_spatial_autocorr.py - Actual implementation
def aggregate_zone_ranges(autocorr_ranges):
    """
    Combine zone-specific range estimates into global summary
    """
    # Filter out failed variogram fits (range=0)
    filt_ranges = [(r, n) for r, n in autocorr_ranges if n > 0]
    
    # Extract ranges and sample sizes from zone results
    sample_sizes = np.array([n for r, n in filt_ranges])  # Data points per zone
    ranges = np.array([r for r, n in filt_ranges])        # Range estimates per zone
    
    # Convert sample sizes to weights (sum to 1.0)
    weights = sample_sizes / sample_sizes.sum()
    
    # Weighted statistics for robust range estimation
    range_stats = {
        "mean": np.average(ranges, weights=weights),        # ← Used for H3 sizing
        "std": np.sqrt(np.average((ranges - ranges.mean())**2, weights=weights)),
        "median": np.median(ranges),                        # Robust central estimate
        "q05": np.quantile(ranges, 0.05),                  # Conservative (5th percentile)
        "q95": np.quantile(ranges, 0.95),                  # Liberal (95th percentile)
        "n": sample_sizes.sum(),                           # Total observations
        "n_chunks": len(ranges)                            # Number of zones analyzed
    }
    
    return range_stats
```

#### Real Example: Chunking → Aggregation → Cross-Validation

**Input**: 4 geographic zones with varying data density and range estimates:

**Zone-Specific Results**:
```python
zone_variogram_results = [
    (1200000, 8453),    # Zone 0_0: North America/Europe, 8,453 plots, 1,200 km range
    (1400000, 3287),    # Zone 1_0: East Asia/Alaska, 3,287 plots, 1,400 km range  
    (920000, 6542),     # Zone 0_1: South America/Africa, 6,542 plots, 920 km range
    (650000, 4891),     # Zone 1_1: Australia/Southern Africa, 4,891 plots, 650 km range
]
```

**Weighted Aggregation Calculation**:
```python
# Step 1: Extract data
ranges = np.array([1200000, 1400000, 920000, 650000])      # Zone ranges (meters)
sample_sizes = np.array([8453, 3287, 6542, 4891])         # Zone data densities

# Step 2: Calculate weights
total_samples = sample_sizes.sum()  # 23,173 total observations
weights = sample_sizes / total_samples
# weights = [0.365, 0.142, 0.283, 0.211]  # North America gets highest weight

# Step 3: Weighted mean range
final_range = np.average(ranges, weights=weights)
# final_range = 1,035,847 meters (1,036 km)

# Step 4: Uncertainty quantification  
range_std = np.sqrt(np.average((ranges - ranges.mean())**2, weights=weights))
# range_std = 287,429 meters (287 km uncertainty)
```

**Final Output for Spatial Cross-Validation**:
```python
trait_range_summary = {
    "trait": "X1080_mean",           # Root length per dry mass
    "mean": 1035847,                 # 1,036 km ← Used for H3 hexagon sizing
    "std": 287429,                   # ±287 km uncertainty
    "median": 1060000,               # Robust central estimate  
    "q05": 689000,                   # Conservative (689 km)
    "q95": 1367000,                  # Liberal (1,367 km)
    "n": 23173,                      # Total sPlot observations across all zones
    "n_chunks": 4                    # Number of geographic zones analyzed
}
```

#### Why Weighted Averaging Matters

**Problem with Simple Mean**: `(1200 + 1400 + 920 + 650) / 4 = 1,043 km`
- Treats sparse zones (3,287 plots) equally with dense zones (8,453 plots)
- Ignores data quality differences between regions

**Benefit of Weighted Mean**: `1,036 km` 
- North America/Europe (36.5% weight) dominates due to high data density
- Sparse regions contribute proportionally to their reliability
- More robust estimate for global cross-validation design

**Pipeline Impact**: The weighted mean range (1,036 km) determines:
1. **H3 resolution selection**: Choose hexagon size ≥ 1,036 km  
2. **Cross-validation folds**: Ensure train/test separation > 1,036 km
3. **Spatial independence**: Guarantee no autocorrelation leakage between folds

---

## 4. Spatial Cross-Validation Design

### 4.1 H3 Hexagonal Spatial Indexing

The pipeline uses **Uber's H3 geospatial indexing system** for spatial clustering:

```python
# From skcv_splits.py and spatial_utils.py
import h3

def acr_to_h3_res(autocorr_range_meters):
    """
    Convert autocorrelation range to appropriate H3 resolution
    """
    # H3 resolution levels and approximate edge lengths
    h3_edge_lengths = {
        0: 1107692.0,   # ~1,108 km (continental scale)
        1: 418676.0,    # ~419 km (country scale) 
        2: 158244.0,    # ~158 km (regional scale)
        3: 59810.0,     # ~60 km (metropolitan scale)
        4: 22606.0,     # ~23 km (city scale)
        # ... continues to resolution 15
    }
    
    # Select resolution where hexagon size >= autocorr range
    for resolution, edge_length in h3_edge_lengths.items():
        if edge_length >= autocorr_range_meters:
            return resolution
    
    return 0  # Default to largest hexagons

def assign_hexagons(df, h3_resolution):
    """
    Assign each data point to H3 hexagon
    """
    def get_hex_id(lat, lon, res):
        return h3.geo_to_h3(lat, lon, res)
    
    # H3 requires geographic coordinates (lat/lon in degrees)
    df["hex_id"] = df.apply(
        lambda row: get_hex_id(row["lat"], row["lon"], h3_resolution), 
        axis=1
    )
    
    return df
```

#### Why H3 Hexagons Over Other Grids?

**Simple analogy**: Imagine trying to tile a bathroom floor. You could use square tiles or hexagonal (6-sided) tiles. Hexagons have some nice properties that make them better for spatial analysis.

**Advantages of hexagonal grids:**
- **Neighbors**: Each hexagon has 6 equidistant neighbors (squares have 4 close + 4 diagonal neighbors at different distances)
- **Shape**: More circular, which better approximates how distance-based clustering actually works in nature
- **Distortion**: Less distortion when mapping the curved Earth onto flat surfaces  
- **Hierarchy**: Each parent hexagon contains exactly 7 children at the next finer resolution (neat mathematical property)
- **Industry standard**: Used by Uber (for ride routing), Meta (for location analytics), and other major geospatial companies

**H3 Global Properties:**
- **Resolution 0**: 122 base hexagons cover the entire Earth
- **Edge length**: ~1,107 km at the largest scale (continental scale)
- **Coverage**: Each hexagon covers ~4.25 million km² (larger than most countries)

**Why this matters**: H3 gives us a consistent, mathematically elegant way to divide the Earth into spatial regions of any size we need, from continental down to city-block level.

### 4.2 Spatial Fold Assignment with Distribution Balancing

The critical innovation is assigning **hexagons** (not individual points) to CV folds:

```python
# From skcv_splits.py
def assign_folds(df, n_splits=5, n_sims=200, trait_col="trait_value"):
    """
    Assign CV folds at hexagon level with distribution balancing
    """
    unique_hexagons = df["hex_id"].unique()
    
    best_similarity = -1
    best_assignment = None
    
    # Try 200 random hexagon→fold assignments
    for simulation in range(n_sims):
        # Randomly shuffle hexagons
        shuffled_hexs = np.random.permutation(unique_hexagons)
        
        # Round-robin assignment to folds
        hex_to_fold = {
            hex_id: i % n_splits 
            for i, hex_id in enumerate(shuffled_hexs)
        }
        
        # Apply hexagon assignments to all data points
        df["fold"] = df["hex_id"].map(hex_to_fold)
        
        # Evaluate trait distribution similarity across folds
        similarity = calculate_similarity_kg(range(n_splits), df, trait_col)
        
        # Keep best assignment (highest similarity)
        if similarity > best_similarity:
            best_similarity = similarity
            best_assignment = df["fold"].copy()
    
    df["fold"] = best_assignment
    return df

def calculate_similarity_kg(folds, df, trait_col):
    """
    Kolmogorov-Smirnov test for fold distribution similarity
    """
    from scipy.stats import ks_2samp
    
    # Compute pairwise KS p-values between all folds
    p_values = []
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            fold_i_values = df[df["fold"] == i][trait_col]
            fold_j_values = df[df["fold"] == j][trait_col]
            
            _, p_value = ks_2samp(fold_i_values, fold_j_values)
            p_values.append(p_value)
    
    # Higher mean p-value = more similar distributions
    return np.mean(p_values)
```

#### Example Fold Assignment Results

**Real example**: For root length trait (X1080_mean) with 946 km autocorrelation range:

**Spatial Setup:**
- **Autocorr range**: 946,462 meters (946 km)
- **H3 resolution**: 0 (largest hexagons ~1,107 km diameter)  
- **Global coverage**: 102 hexagons contain our data points
- **Total observations**: 45,234 vegetation survey plots

**Quality Checks:**
- **Spatial independence**: Points in different folds are >946 km apart (guaranteed by hexagon size)
- **Distribution balance**: KS p-value = 0.847 (high similarity means trait distributions are well-balanced across folds)

**What this means**: When we test Fold 0, the model has never seen data from those 20 hexagons, and they're far enough away (>946 km) that spatial autocorrelation shouldn't help the prediction.

### 4.3 Complete Spatial Cross-Validation Workflow

```python
# From skcv_splits.py - Complete workflow
def create_spatial_cv_splits(trait_col, ranges_df, traits_df, cfg):
    """
    Complete workflow for spatial cross-validation split creation
    """
    # Step 1: Extract trait data with coordinates
    trait_data = traits_df[[trait_col, "x", "y"]].dropna()
    
    # Step 2: Look up autocorrelation range
    trait_range = ranges_df[ranges_df["trait"] == trait_col]["mean"].values[0]
    
    # Step 3: Handle coordinate system conversion
    if cfg.crs == "EPSG:6933":
        # Reproject to geographic coordinates for H3
        trait_data = reproject_xy_to_geo(trait_data, from_crs="EPSG:6933")
        trait_data = trait_data.rename(columns={
            "x": "x_old", "y": "y_old",  # Keep original coordinates
            "lat": "y", "lon": "x"        # Use lat/lon for H3
        })
    
    # Step 4: Assign hexagon IDs
    h3_resolution = acr_to_h3_res(trait_range)
    trait_data = assign_hexagons(trait_data, h3_resolution)
    
    # Step 5: Assign CV folds (hexagon level)
    trait_data = assign_folds(
        trait_data, 
        n_splits=cfg.train.cv_splits.n_splits,      # 5 folds
        n_sims=cfg.train.cv_splits.n_sims,          # 200 simulations
        trait_col=trait_col
    )
    
    # Step 6: Revert to original coordinate system
    if cfg.crs == "EPSG:6933":
        trait_data = trait_data.drop(columns=["x", "y"]).rename(columns={
            "x_old": "x", "y_old": "y"
        })
    
    # Step 7: Save fold assignments
    splits_data = trait_data[["x", "y", "fold"]].drop_duplicates(subset=["x", "y"])
    splits_data.to_parquet(f"data/features/{cfg.PFT}/{cfg.model_res}/skcv_splits/{trait_col}.parquet")
    
    return splits_data
```

---

## 5. Model Training and Evaluation

## 5. Model Architecture and Training

### 5.1 Model Design Principles

**Model Type**: **Gradient-Boosted Decision Trees (GBDTs)** - chosen for their ability to:
- Capture non-linear relationships between traits and environmental predictors
- Handle mixed data types (continuous, categorical, missing values)
- Provide feature importance rankings
- Scale efficiently to large datasets

**Training Strategy**: **Separate models** for each combination of:
- **31 traits** (SLA, leaf nitrogen, wood density, etc.)
- **5 resolutions** (1, 22, 55, 111, 222 km)  
- **3 data subsets**:
  - **SCI**: sPlot vegetation surveys only (high quality, limited coverage)
  - **CIT**: Citizen science/GBIF only (broader coverage, more noise)
  - **COMB**: Combined dataset with down-weighting of noisy citizen science

**Total Models**: 31 traits × 5 resolutions × 3 data types = **465 separate models**

**Input Structure**:
- **Predictors**: ~150 environmental variables (always the same across models)
  - MODIS reflectance bands
  - WorldClim bioclimatic variables  
  - SoilGrids soil properties
  - VODCA vegetation optical depth
  - Global canopy height metrics
- **Targets**: Community-weighted mean (CWM) traits per grid cell

### 5.2 Gradient Boosted Decision Trees (AutoGluon)

```python
# From train_models.py (simplified)
from autogluon.tabular import TabularPredictor

def train_trait_model(trait_col, splits_data, feature_data, cfg):
    """
    Train GBDT model with spatial cross-validation
    """
    # Combine features and targets
    model_data = feature_data.merge(splits_data, on=["x", "y"])
    
    # Configure AutoGluon
    predictor = TabularPredictor(
        label=trait_col,
        problem_type="regression",
        eval_metric="pearson"
    ).fit(
        train_data=model_data,
        presets="high_quality",
        time_limit=7200,  # 2 hours per trait
        num_gpus=2,
        included_model_types=["GBM"],  # Gradient boosting only
        cv_fit=True,
        custom_cv=create_spatial_cv_splitter(splits_data)
    )
    
    return predictor

def create_spatial_cv_splitter(splits_data):
    """
    Create spatial CV splitter for AutoGluon
    """
    def spatial_cv_split():
        for fold_id in range(5):
            train_idx = splits_data[splits_data["fold"] != fold_id].index
            test_idx = splits_data[splits_data["fold"] == fold_id].index
            yield train_idx, test_idx
    
    return spatial_cv_split
```

### 5.2 Performance Metrics

**How do we measure if the model is working well?** We use several metrics that focus on spatial generalization:

**Primary Metrics:**
- **Pearson correlation (r)**: How well predicted values match observed values (0 = no relationship, 1 = perfect match)
- **RMSE**: Root Mean Squared Error - average magnitude of prediction errors  
- **Normalized RMSE**: RMSE divided by the range of observed values (makes it comparable across different traits)

**Spatial-Specific Metrics:**
- **Spatial correlation**: Correlation specifically in held-out geographic regions (tests true spatial generalization)
- **CV consistency**: How similar the model performance is across different folds (low variation = more reliable model)

**What good performance looks like:**
- **Random CV**: r = 0.85-0.95 (looks great but inflated by spatial autocorrelation)
- **Spatial CV**: r = 0.45-0.75 (more realistic measure of true generalization ability)
- **Performance drop**: Usually 10-30% lower with spatial CV, which reveals the model's true spatial transfer capability

### 5.3 Dealing with Data Heterogeneity and Bias

**The Non-IID Problem**: Ecological data are heavily biased geographically and environmentally:
- **Europe, North America, Japan, Australia**: Overrepresented in surveys and citizen science
- **Tropics, deserts, alpine regions, Central Asia**: Severely underrepresented
- **Consequence**: If using random CV, performance looks artificially high because train/test share similar regions

**Mitigation Strategies**:

1. **Spatial Cross-Validation**: Forces models to predict across different regions (closer to real-world deployment)

2. **Down-weighting Noisy Samples**: In COMB models, citizen science observations are explicitly down-weighted so expert survey data dominate signal quality

3. **Multiple Data Subsets**: By training SCI, CIT, and COMB separately, researchers can evaluate how each data type contributes:
   - **SCI**: High reliability, limited coverage
   - **CIT**: Broad coverage, variable quality  
   - **COMB**: Best of both worlds with careful weighting

4. **Independent Validation**: Using only expert survey data (sPlot) for testing avoids "self-validation" with noisy citizen science inputs

5. **Area of Applicability (AOA)**: Identifies where predictions extrapolate beyond training domain - helps flag unreliable outputs in undersampled regions

**Why Not Random CV?**
Random CV would allow training and test samples to fall very close together geographically. Because ecological and remote sensing data are **spatially autocorrelated**, random splits would **inflate performance metrics** and give false confidence in model generalization.

---

## 6. Uncertainty Quantification

### 6.1 Coefficient of Variation (COV) Maps

**What is COV?** Imagine you predict the same location 5 times using 5 different training sets. If you get very similar answers each time, that's low uncertainty. If the answers vary a lot, that's high uncertainty.

**How COV is computed:**

For each location, we have 5 predictions (one from each CV fold). COV measures their consistency:

$$\text{COV} = \frac{\text{standard deviation}}{\text{mean}}$$

```python
def compute_cov_maps(cv_predictions):
    """Quantify prediction uncertainty across CV folds"""
    cov_map = cv_predictions.groupby(["x", "y"]).agg({
        "prediction": ["mean", "std"]
    })
    
    # COV = standard deviation / mean
    cov_map["cov"] = cov_map[("prediction", "std")] / cov_map[("prediction", "mean")]
    
    return cov_map
```

**Interpreting COV values:**
- **Low COV** (<0.2): Stable predictions, reliable model - "we're confident about this location"
- **Medium COV** (0.2-0.5): Moderate uncertainty - "reasonable confidence but some variation"  
- **High COV** (>0.5): High uncertainty, sparse training data - "we're not very sure about this prediction"

**Where do we see high COV?** Usually in deserts, mountains, and remote areas where we have little training data.

### 6.2 Area of Applicability (AOA)

**What is AOA?** Think of it as asking: "Is this new location similar enough to places the model has seen before?" If yes, the prediction is probably reliable. If no, the model is extrapolating to completely new conditions.

**Simple analogy**: If you learn to drive in sunny California suburbs, you might not be reliable at predicting how to drive in a snowy mountain village. AOA identifies the "snowy mountains" of our trait predictions.

**How AOA is computed:**

1. **For each prediction location**: Calculate how environmentally similar it is to all training locations (using climate, soil, vegetation data)
2. **Find the closest match**: What's the minimum environmental distance to any training location?
3. **Set a threshold**: If you're farther than 90% of training locations are from each other, you're "outside the AOA"

```python
def compute_aoa_maps(training_predictors, prediction_predictors):
    """Identify where predictions extrapolate beyond training domain"""
    from sklearn.metrics import pairwise_distances
    
    # Compute dissimilarity index
    distances = pairwise_distances(
        prediction_predictors, 
        training_predictors, 
        metric="euclidean"
    )
    
    # Minimum distance to training data
    min_distances = distances.min(axis=1)
    
    # Threshold based on training distribution  
    aoa_threshold = np.percentile(min_distances[training_mask], 90)
    aoa_binary = min_distances < aoa_threshold
    
    return {
        "aoa_binary": aoa_binary,           # Inside/outside applicability
        "dissimilarity": min_distances,     # Continuous similarity measure
        "threshold": aoa_threshold          # Applicability threshold
    }
```

**Interpreting AOA:**
- **Inside AOA**: The model has seen similar environmental conditions before - predictions are probably reliable
- **Outside AOA**: The model is extrapolating to new environmental conditions - be cautious about these predictions

**Common outside-AOA areas**: High mountains, deep deserts, arctic regions, and other extreme environments underrepresented in training data.

### 6.3 Detailed AOA Implementation and Interpretation

**AOA Computation Method**: Uses the **Dissimilarity Index (DI)** approach (Meyer & Pebesma 2021):

1. **Training Domain Definition**: Each training grid cell has a vector of predictor values (MODIS, WorldClim, SoilGrids, VODCA, canopy height) forming the "reference set of conditions"

2. **Dissimilarity Calculation**: For every prediction grid cell:
   - Compute **Euclidean distance** in standardized predictor space to all training points
   - Take **minimum distance** as that cell's Dissimilarity Index (DI)

3. **Threshold Determination**: During cross-validation, compare DI distribution to prediction error:
   - **Below threshold** → pixel is *inside AOA* (interpolation, reliable)
   - **Above threshold** → pixel is *outside AOA* (extrapolation, unreliable)

**AOA Results in This Study**:

- **SCI-only training**: Smaller AOA because surveys clustered in Europe/North America
  - Many tropical, desert, alpine environments fall outside AOA
  
- **COMB (SCI + CIT)**: AOA increases for all traits at all resolutions
  - **Average global gain**: +2.4 percentage points
  - **Maximum gain**: +9.2 points (wood fiber length at 1km)
  - **Strongest effect**: At finer resolutions (1km) where CIT adds many observations

**AOA Map Interpretation**:
- **High AOA (low DI)**: Reliable prediction - training domain well covered
- **Low AOA (high DI)**: Extrapolation - training domain not covered  
- **Magenta areas**: Outside AOA for both SCI and COMB (extreme environments)
- **Olive areas**: Unreliable in SCI but reliable in COMB (CIT expanded coverage)

**Why CIT Increases AOA**: 
- CIT data don't change the predictor variables (environmental layers stay the same)
- CIT adds **many more training cells** across wider environmental gradients
- This expands the **envelope of training predictor values**
- Result: More pixels find "similar" training conditions → larger AOA

### 6.4 Coefficient of Variation (COV) Implementation Details

**COV Computation Process**:

1. **Spatial cross-validation setup**: World divided into geographic blocks by resolution
2. **Prediction collection**: Each grid cell predicted multiple times (once per CV fold when held out)
3. **Variation calculation**: For each pixel across CV folds:
   - Compute **mean prediction** ($\mu$)
   - Compute **standard deviation** ($\sigma$)  
   - Calculate $\text{COV} = \frac{\sigma}{\mu}$

**COV Interpretation in Practice**:
- **Low COV**: Predictions stable regardless of training subset → strong, generalizable trait-environment relationships
- **High COV**: Predictions fluctuate with training folds → uncertainty due to sparse/biased training data
- **Common high-COV areas**: Deserts, alpine regions, tropics with few surveys

**Effect of COMB Models**:
- COMB consistently shows **lower COV** than SCI-only models
- Adding CIT reduces data sparsity and increases geographic coverage
- Results in more stable predictions across different training scenarios

---

## 7. Final Products and Outputs

### 7.1 Global Trait Maps

**What do we actually produce?** The pipeline creates three main types of outputs:

**Primary Trait Maps:**
- **Format**: GeoTIFF rasters (standard geographic image files that work in any GIS software)
- **Resolutions**: 1km, 22km, 55km, 111km, 222km (from local to continental scales)
- **Traits**: 31 functional traits (leaf area, wood density, root depth, etc.)
- **Data Types**: 
  - SCI (expert surveys only) 
  - CIT (citizen science only)
  - COMB (combined - usually the best version)
- **Coverage**: Global, land areas only (oceans excluded)

**Uncertainty Layers:**
- **COV maps**: How consistent are the predictions? (low values = more reliable)
- **AOA maps**: Where is the model extrapolating? (binary "reliable/unreliable" + continuous similarity scores)
- **Coverage**: Same global extent as the trait maps

**Evaluation Results:**
- **Performance tables**: How well does each trait model work? (correlation, error rates by trait and resolution)
- **Spatial validation**: How well do models predict in completely new regions?
- **Feature importance**: Which environmental variables matter most for each trait?

### 7.3 Detailed Evaluation Pipeline

**Model Training Setup**: For comprehensive assessment, the pipeline trains:
- **31 traits** × **5 resolutions** × **3 data subsets** = **465 separate GBDT models**
- **Predictors**: ~150 environmental variables (static for all models)
- **Targets**: Community-weighted mean (CWM) traits (varies by SCI/CIT/COMB)

**Cross-Validation Strategy**: **Spatial K-fold cross-validation** (not random splits)
- Training and test data split by **geographic blocks**
- Test cells are spatially independent of training cells
- Avoids overestimating performance due to spatial autocorrelation
- **Evaluation metrics**: Correlation coefficient (r) + normalized RMSE (nRMSE)

**Validation Data Source**: Independent test set from **held-out sPlot vegetation surveys**
- Ensures consistent, systematic benchmark across SCI, CIT, and COMB models
- Avoids "self-validation" with potentially noisy citizen science inputs
- Provides stable, unbiased performance assessment

**Geographic Block Cross-Validation Design**:
- **Unit of split**: Grid cells at chosen modeling resolution (1, 22, 55, 111, 222 km)
- **Blocking method**: Cells grouped into **spatially contiguous geographic blocks**
  - Systematic tiling of global grid (not arbitrary splits)
  - Resembles "checkerboard" partitioning with alternating train/test blocks
- **Block sizes vary by resolution**:
  - **1 km**: Blocks cover tens to hundreds of km across
  - **111-222 km**: Blocks can be country or sub-continental sized (Spain, Madagascar, Eastern US)
- **No biome stratification**: Folds are purely geographic, not ecological
- **Intentional challenge**: Some test folds contain environments missing in training (tests true out-of-region generalization)

**Benefits of Geographic Block CV**:
- Prevents **spatial leakage** (nearby training/test points artificially boosting performance)
- Tests **spatial transferability**: Can model trained in one region generalize to another?
- Reveals regional biases (e.g., Portugal with dense citizen science but few surveys)
- Produces realistic performance estimates for global mapping tasks

**Cross-Validation Rotation Logic**: K-fold spatial CV with rotating held-out geographic blocks
- Each region of the world serves once as test data while rest used for training
- Ensures evaluation reflects global spatial generalization rather than local autocorrelation
- **Example with K=5 folds**:
  - Round 1: Train on folds 2-5, test on fold 1
  - Round 2: Train on folds 1,3-5, test on fold 2
  - Continue until fold 5 has been tested
  - Every region contributes to both training and independent testing

### 7.2 Spatial Cross-Validation Outputs

**What technical files does the pipeline create for researchers?**

**Cross-Validation Split Files:**
- **Location**: `data/features/{PFT}/{resolution}/skcv_splits/`
- **Format**: One parquet file per trait (e.g., `X1080_mean.parquet`)
- **Contents**: For each trait, tells you which CV fold each location belongs to
  - `x`: EPSG:6933 x-coordinates (meters)
  - `y`: EPSG:6933 y-coordinates (meters) 
  - `fold`: CV fold assignment (0-4)
- **Size**: ~45,000 locations × 3 columns per trait

**Spatial Autocorrelation Summary:**
- **Location**: `reference/spatial_autocorr_{PFT}_{resolution}_mean.parquet`
- **Purpose**: Lookup table that stores the spatial range for each trait
- **Contents**:
  - `trait`: Trait identifier (e.g., X1080_mean)
  - `mean`: Weighted mean autocorr range (meters) - the key number for CV splits
  - `std`: How much the range varies across regions
  - `median`: Robust central estimate (less affected by outliers)
  - `q05`: Conservative estimate (5th percentile)
  - `q95`: Liberal estimate (95th percentile)
  - `n`: Total samples used in analysis
  - `n_chunks`: Number of geographic regions analyzed

**Why these files matter**: They allow other researchers to reproduce the exact same spatial cross-validation setup and understand the spatial structure of each trait.

---

## 8. Motivation / Why multi-target modelling

## 9. Key Innovations and Contributions

### 9.1 Methodological Advances

**What makes this pipeline different from standard approaches?**

**Trait-Specific Spatial Cross-Validation:**
- **Innovation**: Each trait gets its own custom autocorrelation range and hexagon size
- **Why better**: Some traits (like leaf size) vary over short distances, others (like wood density) are similar across vast regions
- **Traditional approach**: Uses the same fixed geographic blocks for all traits (not optimal)

**H3 Hexagonal Clustering:**
- **Innovation**: Global consistent hexagonal spatial indexing system
- **Benefits**: 
  - No arbitrary grid placement decisions
  - Hierarchical structure (big hexagons contain smaller ones)
  - Works seamlessly from local neighborhood to global scales
- **Industry standard**: Same system used by Uber, Meta, and other major tech companies

**Distribution-Balanced Folds:**
- **Innovation**: Uses Kolmogorov-Smirnov test optimization to ensure trait distributions are similar across folds
- **Why important**: Achieves both spatial independence AND statistical balance
- **Robustness**: Tries 200 different random assignments to find the best one

**Multi-Resolution Training:**
- **Innovation**: Train completely separate models at each spatial scale (1km, 22km, 55km, etc.)
- **Why better**: Avoids scale-transfer bias from naive upscaling
- **Ecological relevance**: Captures scale-dependent ecological processes (local competition vs. regional climate)

### 9.2 Practical Impact

**How will this research be used in the real world?**

**Global Trait Mapping:**
- **Coverage**: First wall-to-wall global trait maps at multiple spatial scales
- **Uncertainty quantification**: Every pixel comes with reliability estimates (COV, AOA)
- **Accessibility**: Standard GeoTIFF format works in any GIS software or Google Earth Engine

**Biodiversity Monitoring:**
- **Baseline establishment**: Creates global trait baselines for detecting future changes
- **Scale flexibility**: Same methodology works for local conservation projects and global assessments
- **Data integration**: Compatible with satellite monitoring and climate datasets

**Climate Research Applications:**
- **Earth system models**: Trait maps help parameterize how vegetation responds to climate change
- **Carbon cycle research**: Plant traits directly affect how much carbon is stored and cycled through ecosystems
- **Climate adaptation**: Understanding trait patterns helps predict which species will survive climate change

**Simple examples**: 
- Conservation groups can identify functionally unique ecosystems that need protection
- Climate scientists can improve predictions of forest response to warming
- Agricultural researchers can understand crop trait patterns across environmental gradients

---

## 10. Code Implementation Summary

### 10.1 Key Scripts and Functions

**Data Processing Pipeline**:
- **`harmonize_eo_data.py`**: Reproject and resample Earth observation datasets to common grid
- **`match_gbif_pfts.py`**: Filter and match GBIF species to plant functional types (PFTs)
- **`build_gbif_maps.py`**: Create frequency-weighted trait maps from GBIF occurrence data
- **`build_splot_maps.py`**: Create community-weighted trait maps from sPlot vegetation surveys
- **`build_try_traits.py`**: Load, filter, and process TRY trait data, aggregate by species/PFT
- **`extract_splot.py`**: Extract and preprocess sPlot data for mapping and analysis
- **`mask.py`**: Apply spatial masks to restrict analysis to study area
- **`subsample_gbif.py`**: Randomly subsample GBIF data to reduce size or balance samples

**Spatial Autocorrelation Analysis**:
- **`calc_spatial_autocorr.py`**: Variogram analysis and range estimation using PyKrige

**Cross-Validation Splits**:
- **`skcv_splits.py`**: H3 hexagon assignment and fold creation with KS-test optimization

**Model Training and Evaluation**:
- **`train_models.py`**: AutoGluon GBDT training with spatial cross-validation
- **`back_transform_predict.py`**: Convert model predictions from transformed space back to original units

**Uncertainty Analysis**:
- **`aoa.py`**: Area of applicability computation using dissimilarity index
- **`cov.py`**: Coefficient of variation mapping across CV folds

**Final Product Generation**:
- **`build_final_product.py`**: Assemble final output datasets from processed results
- **`build_final_metadata.py`**: Compile and write metadata for final data products
- **`reproject_final_maps_for_web.py`**: Reproject final rasters for web visualization
- **`consolidate_biomes.py`**: Merge biome data from different sources into unified format
- **`standardize_other_products.py`**: Harmonize external trait/EO products for integration

**Data Management and Distribution**:
- **`push_sftp.py`**: Upload processed data products to remote server via SFTP
- **`xfer_gs_assets_to_gee.py`**: Transfer GeoSense assets to Google Earth Engine

**Functional Diversity Metrics**:
Additional outputs beyond mean trait values include:
- **Functional Richness (f_ric)**: Range of trait values present (trait space filled)
- **Functional Evenness (f_eve)**: How evenly trait values are distributed among species  
- **Functional Divergence (f_div)**: How far trait values are from center of trait space
- **Functional Redundancy (f_red)**: Overlap in trait values among species
- **Species Richness (sp_ric)**: Simple species count per grid cell
- **Functional Richness SES (f_ric_ses)**: Standardized effect size for functional richness

**Quality Control Procedures**:
- **Subsampling rationale**: If grid cell has >max_count observations, randomly select max_count to:
  - Avoid bias from cells with many observations dominating analyses
  - Ensure comparability across grid cells (similar sample sizes)
  - Reduce computation time while preserving information
  - Prevent overrepresentation due to uneven sampling effort

### 10.2 Configuration Management

**How is the pipeline configured?** Everything is controlled through a YAML configuration file (`params.yaml`):

**Basic Settings:**
- **PFT**: "Shrub_Tree_Grass" (plant functional type filter)
- **model_res**: "55km" (target spatial resolution)  
- **crs**: "EPSG:6933" (coordinate system - Equal Area Cylindrical projection)
- **target_resolution**: 55000 (grid cell size in meters)

**Cross-Validation Settings:**
- **n_splits**: 5 (number of CV folds)
- **n_sims**: 200 (fold optimization iterations - tries 200 random assignments)
- **range_stat**: "mean" (uses mean autocorrelation range for hexagon sizing)

**Trait Configuration:**
- **traits**: [4, 6, 13, 14, 15, 21, 26, 27, 46, 47, 50, 55, 78, 95, 138, 144, 145, 146, 163, 169, 237, 281, 282, 289, 297, 614, 1080, 3106, 3113, 3117, 3120] (TRY database trait IDs)
- **trait_stats**: ["mean", "std", "median", "q05", "q95", "count"] (statistical summaries computed for each trait)

**What this means**: Changing these parameters lets you run the pipeline for different plant types (trees vs. grasses), different spatial scales (local vs. global), or different trait sets without modifying the code.

### 10.3 Detailed Data Flow Implementation

**Pipeline Implementation**: Raw → Interim → Features → Processed → Models → Results

#### GBIF Processing Flow:
```
raw/gbif/*.dvc
→ Load raw GBIF data (parquet format)
→ Filter by Plant Functional Type (PFT) using config (e.g., Shrub/Tree/Grass only)
→ Set species name as index
→ Optional subsampling: randomly select fraction of observations (e.g., 1%) for testing/balancing
→ Join with TRY trait data (species-level matching)
→ Coordinate reprojection: EPSG:4326 → EPSG:6933 (Equal Area) if needed
→ Spatial binning: assign grid cell (row, col) using spatial transform
  - Grid resolution from config (1km, 55km, etc.)
  - Each observation mapped to grid cell based on x/y coordinates
→ Masking/filtering:
  - Remove observations outside study area (using mask raster/bounding box)
  - Filter by min/max count per cell (e.g., min_count=10, max_count=500)
→ Aggregate per grid cell:
  - Compute mean trait values, observation counts, etc.
  - If >max_count, randomly subsample to max_count
→ Output: gridded GBIF data with trait information
```

#### TRY Trait Processing Flow:
```
raw/TRY_*.dvc, try_pft_*.csv.dvc, TRY_trait_table_*.txt.dvc
→ Load raw trait tables (parquet, csv, txt formats)
→ Filter for relevant traits (from configuration)
→ Optional: PCA on trait matrix if configured (dimensionality reduction)
→ Map species to PFTs using lookup table
→ Aggregate trait values per species:
  - Compute mean, median, or other summary statistics per trait
  - Remove species with missing or insufficient data
→ Output: trait table ready for joining with occurrence data
```

#### Environmental Data Processing Flow:
```
raw/esa_worldcover_*.dvc, ETH_GlobalCanopyHeight_*.dvc, 
modis_sur_refl_*.dvc, soilgrids_*.dvc, vodca_*.dvc, wc2-1_30s_bio.dvc
→ Load raster Earth Observation datasets
→ Reproject to target CRS (EPSG:6933)
→ Resample to target grid resolution using nearest neighbor or bilinear interpolation
→ Apply masking:
  - Use study area mask (biomes.tif or bounding box)
  - Set values outside mask to NaN or ignore
→ Stack features:
  - Extract EO values per grid cell (canopy height, soil, climate)
  - Optionally aggregate (mean, min, max) if multiple rasters per cell
→ Output: EO feature grid ready for modeling
```

#### sPlot Processing Flow:
```
raw/splot4-0.dvc, sPlotOpen_v76.dvc
→ Load plot-level vegetation survey data
→ Filter for relevant plots/species (by region, PFT)
→ Map to grid cells using coordinates and grid transform
→ Aggregate statistics per cell:
  - Compute mean, count, summary stats for each plot variable
→ Apply masking: remove plots outside study area
→ Output: gridded plot data
```

#### Final Integration Flow:
```
(GBIF + TRY traits + EO data + sPlot)
→ Join GBIF observations with trait data (by species)
→ Spatial binning: assign each observation to grid cell
→ For each grid cell:
  - Aggregate trait values (mean, count, etc.)
  - Aggregate EO features (mean, min, max, etc.)  
  - Aggregate plot-level data if available
→ Apply mask to exclude cells outside study area
→ Output: Feature matrix per grid cell (ready for modeling)

Final Model Inputs & Outputs:
- Inputs: EO predictor stack (~150 variables)
- Targets: Trait CWMs per grid (SCI, CIT, COMB)
- Outputs: Global trait maps (GeoTIFFs, 1-222 km), COV maps, AOA masks
```

#### Key Implementation Details:

**Coordinate System Handling**:
- All data aligned to **EASE-Grid v2.0 (EPSG:6933)** for consistent processing
- GBIF+TRY yield **frequency means**, sPlot+TRY yield **true CWMs**
- Always aggregate **directly from raw records** for each target grid (no upscaling from 1km)

**Data Type Distinctions**:
- **sPlot**: Community-weighted mean at plot level, weighted by relative cover/abundance (continuous measure)
- **GBIF**: Frequency-weighted mean at grid level, weighted by occurrence record counts (discrete, often biased by sampling effort)

**Resolution Independence**: Each target resolution (1, 22, 55, 111, 222 km) processes data independently from raw sources, avoiding scale-transfer artifacts from naive upscaling.

---

## Conclusion

This pipeline represents a comprehensive approach to global ecological modeling that addresses fundamental challenges in spatial data analysis. The combination of **variogram-based spatial autocorrelation analysis**, **H3 hexagonal spatial indexing**, and **distribution-balanced cross-validation** provides a robust framework for creating reliable global trait maps.

**Key Takeaways:**
1. **Spatial autocorrelation must be quantified and accounted for** in ecological machine learning
2. **Trait-specific spatial ranges** enable optimized cross-validation design  
3. **Hexagonal grids provide superior spatial clustering** compared to arbitrary rectangular blocks
4. **Multi-resolution modeling** captures scale-dependent ecological processes
5. **Uncertainty quantification** is essential for responsible global predictions

The methodology is **generalizable beyond plant traits** to any spatially autocorrelated ecological or environmental variable, making it valuable for the broader geospatial modeling community.


## Extended Glossary: Data Sources

### TRY

- **Origin**: Initiated by the Max Planck Institute for Biogeochemistry and iDiv, Halle-Jena-Leipzig.
- **Content**: A global archive of *plant trait measurements* (e.g. leaf area, seed mass, wood density) collected from published studies and researcher contributions.
- **Scale**: Tens of millions of trait records covering >400,000 species, but strongly biased toward Europe and temperate regions.
- **Resolution**: Point-based trait measurements; no inherent spatial resolution — must be linked with occurrence data.
- **Temporal coverage**: Collected over decades; trait data are essentially static (traits measured once, not monitored through time).
- So, a “point” in TRY is **not a record of species presence/absence or abundance**, but rather “at this place and time, someone measured this trait on one or more individuals of species X.

### GBIF (Global Biodiversity Information Facility)

- **Origin**: International open-data infrastructure funded by governments worldwide.
- **Content**: Species occurrence records from *citizen science platforms* (iNaturalist, Pl@ntNet, etc.), herbaria, museums, monitoring networks, and research projects.
- **Scale**: >2 billion occurrence records across all taxa; >500 million plant records alone.
- **Resolution**: Coordinates vary from precise GPS (few meters) to generalized grid cells (up to ~10 km).
- **Temporal coverage**: Ongoing, with strong growth since 2000 due to citizen science apps; includes historical herbarium specimens going back centuries.

### sPlot (Global Vegetation-Plot Database)

- **Origin**: International initiative hosted by iDiv, collating vegetation survey data from hundreds of regional and national databases.
- **Content**: *Vegetation plots* with species composition, abundance/cover, and environmental metadata.
- **Scale**: >2 million plots, covering all major biomes but with clustering in Europe and North America.
- **Resolution**: Plot sizes vary (commonly 10–400 m², but can be up to 1 ha). Each plot has exact geographic coordinates.
- **Temporal coverage**: Surveys conducted from the 1920s to present, but uneven — many plots are single snapshots, not permanent monitoring.

### BIEN (Botanical Information and Ecology Network)

- **Origin**: Collaboration between US-based institutions (Columbia, Yale, Duke, iDiv, etc.).
- **Content**: Aggregates plant distribution records from herbaria, ecological surveys, forest inventories, and trait datasets.
- **Scale**: Hundreds of millions of plant records, heavily covering the Americas. (Strong bias)
- **Resolution**: Mixed — from individual plot inventories to regional aggregations.
- **Temporal coverage**: Historical (herbaria) plus modern records, but not a standardized monitoring program.

### Remote-Sensing / Environmental Predictors

- **MODIS (Moderate Resolution Imaging Spectroradiometer)**: Global surface reflectance at 250–500 m resolution, daily to 16-day composites, continuous since 2000.
- **WorldClim**: Bioclimatic variables (temperature, precipitation) at ~1 km² spatial resolution, based on interpolated weather station data (1950–2000 normals, with future projections).
- **SoilGrids**: Global gridded soil properties (texture, organic matter, nutrients) at ~250 m spatial resolution, modeled from 100,000+ soil profiles and covariates.
- **VODCA (Vegetation Optical Depth Climate Archive)**: Global microwave-based vegetation water content, at ~25 km resolution, spanning 1987–2016.
- **Global canopy height (GEDI, ICESat, etc.)**: Derived from LiDAR, ~1 km to 25 m resolution depending on product, available since 2019.

### What MODIS is (in the context of this study)

- The study uses **MODIS surface reflectance**—optical measurements of the Earth’s surface in the visible and infrared parts of the spectrum. These data are well suited for vegetation/trait inference because canopy optical properties encode leaf chemistry and structure.
- Concretely, they use the **MODIS/Terra MOD09GA v061 Surface Reflectance Daily L2G** product (native grids at **1 km and 500 m**). MODIS was chosen for its **reliable, continuous, large-scale coverage** compared with other optical missions.

---

### How MODIS is included in the workflow

- **Predictor set:** MODIS surface reflectance is one of the major environmental predictor groups, alongside WorldClim climate variables, SoilGrids soil properties, vegetation optical depth, and canopy height. Across all predictors (~**150** in total; ~**19 billion** observations), **MODIS reflectance is the single most influential set on average** for predicting traits.
- **Trait sensitivity:** Some traits respond particularly strongly to optical reflectance. For example, **specific leaf area (SLA)** shows strong influence from **MODIS reflectance**; structural traits (e.g., plant height, leaf width) lean more on canopy height and vegetation optical depth, and several other traits lean on climate/soil.
- **Why optical data help:** Fine-scale variability in traits is often better captured by **optical remote sensing**, complementing broader climatic/edaphic (soil) gradients.

---

### Spatial & temporal handling of MODIS for this study

- **Native data & aggregation:** Although MODIS MOD09GA provides **daily** surface reflectance at **500 m / 1 km**, the study does **not** build a time series of trait maps. Instead, MODIS reflectance (and other predictors) are **aggregated to the modeling resolutions** and used as static predictors for each trait model. The team trains separate models at **1, 22, 55, 111, and 222 km** rather than upscaling from a single resolution, to avoid scale biases.
- **Grid/projection alignment:** Species/trait reference data (GBIF, sPlot) are projected to **EASE-Grid v2.0, EPSG:6933** to match gridded predictors (including MODIS). Outputs (inference and uncertainty) are stored as cloud-optimized GeoTIFFs.
- **Why not time-dynamic traits (yet):** The authors note that the **high temporal resolution of MODIS** (and other sensors) could enable **multi-temporal trait maps** in the future, but trait measurements themselves are temporally sparse, which limits time-varying trait modeling for now.

---

## Vegetation Surveys in Detail

The paper distinguishes between **scientific vegetation surveys (SCI)** and **citizen science observations (CIT)**. Within SCI, multiple survey traditions are represented:

- **European Vegetation Archive (EVA)** – Large federation of European plot databases; very dense coverage in Central and Eastern Europe.
- **Forest inventory plots** – National or regional forest monitoring networks (e.g. Portugal, Scandinavia, North America). These provide tree species abundance, basal area, biomass.
- **Regional vegetation plot databases** – Africa, South America, and Asia contribute smaller datasets, though often with limited coverage.
- **Permanent plots** – Some areas (e.g. tropical forest monitoring in Amazonia or Borneo) have long-term vegetation plots with repeated censuses, but these are rare compared to one-off plots.

**Characteristics of vegetation surveys**:

---

- Structured and standardized (species presence/abundance measured within a defined area).
- Provide essential *community composition* data that can be linked to trait databases to derive community-weighted means (CWMs).
- Strong in Europe, North America, parts of South America; sparse in Africa, Central Asia, and boreal/tropical regions.
- Plot sizes, protocols, and taxonomic resolution vary among contributing databases, though harmonization (e.g. via sPlot) reduces inconsistencies.