# Complete Spatial Trait Mapp

### 2.2 Spatial Aggregation

Observations are harmonized to a common grid resolution (1–222 km). Each grid cell represents a "community" of species for which a **community-weighted mean (CWM)** trait value is calculated:

$$\text{CWM}_{\text{cell}} = \sum_i p_i t_i$$

where  
- $p_i$ = relative abundance (sPlot) or occurrence frequency (GBIF) of species *i*,  
- $t_i$ = trait value from TRY.

**Key Processing Steps:**

1. **Reprojection**: Convert all datasets to EPSG:6933 (World Equidistant Cylindrical)
   - Preserves areas globally for accurate spatial analysis
   - Enables metric distance calculations essential for spatial autocorrelation
2. **Resampling**: Aggregate/disaggregate to target resolution
   - Conservative resampling (mean, median) for continuous variables
   - Nearest neighbor for categorical data
3. **Quality Control**: Subsample dense cells to avoid bias (e.g., max 500 observations per cell)
4. **Masking**: Apply study area boundaries using ESA WorldCover land use data

```python
# Example: compute CWM for each grid cell
def compute_splot_cwms(plot_data, trait_data):
    """Compute true community-weighted means from plot surveys"""
    cwms_per_plot = []
    
    for plot_id, plot in plot_data.groupby("PlotObservationID"):
        species_abundances = plot[["Species", "Cover_perc"]].set_index("Species")
        plot_traits = species_abundances.join(trait_data, how="inner")
        
        # Compute CWM: weighted by relative abundance/cover
        for trait in trait_columns:
            if len(plot_traits[trait].dropna()) > 0:
                cwm_value = np.average(
                    plot_traits[trait].dropna(),
                    weights=plot_traits.loc[plot_traits[trait].notna(), "Cover_perc"]
                )
                cwms_per_plot.append({
                    "PlotID": plot_id, "trait": trait, "CWM": cwm_value,
                    "x": plot["x"].iloc[0], "y": plot["y"].iloc[0]
                })
    
    return pd.DataFrame(cwms_per_plot)
```

**Key Difference Between Data Sources:**
- **sPlot**: Weighted by actual species abundance/cover in vegetation surveys (true CWMs)
- **GBIF**: Weighted by observation frequency - biased by sampling effort (frequency-weighted means)ating ecological theory, spatial statistics, and machine learning for global trait prediction.*

---

## 1. Overview

This pipeline describes how **plant trait data** from field surveys and citizen science observations are combined with **environmental predictors** and processed into **spatially aware machine learning models**.  
It is based on the framework developed by Lunsk *et al.* (2025) for global plant trait mapping and integrates spatial autocorrelation analysis, H3-based spatial partitioning, and cross-validation.

**Key Innovation**: The pipeline addresses spatial autocorrelation in ecological data through variogram-based spatial cross-validation, ensuring robust model evaluation for global trait mapping applications.

**Pipeline Architecture**: Raw Data → Interim Processing → Feature Engineering → Model Training → Spatial Cross-Validation → Final Products

**Target Resolutions**: Models are trained independently at multiple spatial resolutions (1km, 22km, 55km, 111km, 222km) to avoid scale-transfer bias and capture scale-dependent ecological processes.

---

## 2. Data Inputs and Preprocessing

### 2.1 Core Datasets

| Source | Type | Description |
|--------|------|-------------|
| **sPlot** | Vegetation survey plots | Expert-curated vegetation records with species cover/abundance. |
| **GBIF** | Citizen science occurrences | Opportunistic species presence-only data. |
| **TRY** | Trait measurements | Species-level plant functional traits (e.g., SLA, leaf N). |
| **Environmental Layers** | Predictors | MODIS reflectance, WorldClim, SoilGrids, VODCA, canopy height. |

### 2.2 Spatial Aggregation

Observations are harmonized to a common grid resolution (1–222 km). Each grid cell represents a “community” of species for which a **community-weighted mean (CWM)** trait value is calculated:

\[
\text{CWM}_{\text{cell}} = \sum_i p_i t_i
\]

where  
- \(p_i\) = relative abundance (sPlot) or occurrence frequency (GBIF) of species *i*,  
- \(t_i\) = trait value from TRY.

```python
# Example: compute CWM for each grid cell
for cell in grid_cells:
    species = gbif_data[cell]
    weights = species.abundance / species.abundance.sum()
    traits = try_traits.loc[species.index]
    cwm[cell] = (weights * traits).sum()
```

### 2.3 Environmental Resampling

Each predictor dataset is resampled to the same modeling grid. For example:

- **MODIS** (500m–1km native) → mean/bilinear resampling over pixels within grid cell  
- **WorldClim** (~1km native) → bioclimatic variables aggregated to target resolution
- **SoilGrids** (250m native) → mean or median aggregation for soil properties
- **VODCA** (~25km native) → vegetation optical depth rescaled to match grids
- **ETH Global Canopy Height** (~1km native) → forest structure metrics aggregated

Each grid cell thus receives a feature vector of ~150 predictor variables.

**Coordinate System Requirements:**
- **EPSG:6933** (Equal Area Cylindrical): Primary CRS for global analysis
  - Equal-area projection preserves relative sizes globally
  - Metric coordinates enable direct distance calculations
  - Single CRS for worldwide datasets (no zone boundaries)
- **EPSG:4326** (WGS84): Used only for H3 hexagon assignment
- **UTM zones**: Used for local variogram computation when fine resolution analysis needed

---

## 3. Spatial Autocorrelation Analysis

Spatial autocorrelation quantifies how similar trait values are across space — a fundamental property of ecological data.

### 3.1 Motivation

Because nearby observations are more similar than distant ones, **random cross-validation** violates the IID assumption.  
Hence, we first determine the **spatial range of dependence** for each trait.

### 3.2 Variogram Computation

Data are projected into metric coordinate systems (UTM zones or EPSG:6933) to ensure accurate distance computation.

For each trait *z* within each spatial zone, the **empirical semivariance** is computed:

$$\gamma(h) = \frac{1}{2N(h)} \sum (z_i - z_j)^2$$

where $\gamma(h)$ is the semivariance at lag distance $h$, and $N(h)$ is the number of point pairs at distance $h$.

A **spherical variogram model** is then fitted:

$$\gamma(h) = \begin{cases}
C_0 + (C - C_0)\left[\frac{3h}{2a} - \frac{1}{2}\left(\frac{h}{a}\right)^3\right] & \text{if } h \leq a \\
C_0 + (C - C_0) & \text{if } h > a
\end{cases}$$

with parameters:
- **$C_0$ (nugget)** — microscale noise or measurement error  
- **$C$ (sill)** — total spatial variance  
- **$a$ (range)** — distance beyond which spatial dependence vanishes

**Physical Interpretation:**
- **Short distances** ($h < a$): Points are spatially correlated
- **Long distances** ($h > a$): Points are spatially independent
- **Range parameter** ($a$): Critical distance for spatial cross-validation design

```python
# Implementation using PyKrige
from pykrige.ok import OrdinaryKriging

def calculate_variogram_pykrige(group, trait_col, **kwargs):
    """Fit spherical variogram model to estimate spatial range"""
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

### 3.3 Spatial Chunking for Non-Stationarity

Global datasets exhibit **spatial non-stationarity** - autocorrelation structure varies by region. The pipeline accounts for this by:

1. **Creating spatial zones**: Divide global data into 16 geographic chunks (4×4 grid)
2. **Computing regional variograms**: Separate variogram analysis for each zone with sufficient data (>200 points)
3. **Weighted aggregation**: Combine regional estimates using sample-size weighting

**Example Regional Variation:**
- **North America West**: 1,200 km range (large homogeneous regions)
- **Europe Central**: 850 km range (fragmented landscapes)  
- **Asia Boreal**: 1,400 km range (vast continuous forests)
- **Australia Arid**: 650 km range (strong environmental gradients)
- **Africa Tropical**: 920 km range (climate-driven patterns)

### 3.3 Aggregation Across Zones

Each trait’s autocorrelation ranges (in km) are aggregated across zones:

| Trait | Mean Range (km) | Std | q05 | q95 |
|:------|----------------:|----:|----:|----:|
| SLA | 946 | 234 | 456 | 1234 |
| Leaf N | 820 | 200 | 430 | 1170 |

These ranges guide the **spatial binning scale** for cross-validation.

---

## 4. Defining Spatial Folds Using H3 Hexagonal Grid

### 4.1 Concept

We aim to ensure that training and test data are spatially independent.  
This is achieved by grouping samples into **hexagonal spatial bins** using the [H3](https://eng.uber.com/h3/) indexing system.

Each H3 resolution corresponds to a hexagon size.  
The chosen resolution is matched to the **autocorrelation range** of the trait (rounded to the nearest larger H3 scale).

| Trait | Mean Range (km) | H3 Resolution | Hex Diameter (km) |
|:------|----------------:|---------------:|------------------:|
| SLA | 946 | 0 | ~1107 |
| Leaf N | 820 | 1 | ~550 |

### 4.2 Assigning Observations to Hexagons

Each grid cell is assigned a unique hexagon ID:

```python
import h3
df['hex_id'] = df.apply(lambda r: h3.geo_to_h3(r.lat, r.lon, resolution), axis=1)
```

All points within the same hex share one `hex_id` — treated as one spatial unit in cross-validation.

### 4.3 Creating Balanced Spatial Folds

We assign folds at the **hexagon level**, ensuring each fold is spatially disjoint but statistically similar.

1. Randomly assign hexes to folds (K = 5).  
2. Evaluate balance in trait distributions across folds using a **Kolmogorov–Smirnov (KS) test**.  
3. Keep the assignment with the highest mean p-value (most similar distributions).

```python
from scipy.stats import ks_2samp

best_split = None
best_score = -1
for candidate in random_splits(hex_ids, k=5, n_iter=1000):
    pvals = [ks_2samp(fold_a.trait, fold_b.trait).pvalue for all_fold_pairs]
    score = np.mean(pvals)
    if score > best_score:
        best_split = candidate
        best_score = score
```

Result: folds that are **spatially independent and statistically balanced.**

---

## 5. Cross-Validation Procedure

### 5.1 Fold Rotation

Spatial K-fold cross-validation ensures that each region serves once as test data.

| Round | Train Folds | Test Fold |
|-------:|--------------|-----------|
| 1 | 2–5 | 1 |
| 2 | 1,3–5 | 2 |
| 3 | 1–2,4–5 | 3 |
| 4 | 1–3,5 | 4 |
| 5 | 1–4 | 5 |

At each iteration:
- Train model on K–1 folds (hexes).  
- Test on the held-out fold.  
- Aggregate performance metrics: **Pearson r**, **normalized RMSE (nRMSE)**.

### 5.2 Example Workflow

```python
for fold in folds:
    train = data[data.fold != fold]
    test  = data[data.fold == fold]

    model = GradientBoostedTrees().fit(train.X, train.y)
    preds = model.predict(test.X)

    r = pearsonr(preds, test.y)
    rmse = np.sqrt(((preds - test.y)**2).mean()) / np.mean(test.y)
    results.append((fold, r, rmse))
```

Performance is then averaged across folds.  
This rotation ensures **each geographic block contributes once to testing.**

---

## 6. Model Training and Evaluation

### 6.1 Models

- **Gradient Boosted Decision Trees (GBDT)** — primary model type.  
- Separate models are trained per trait × resolution × data subset (SCI, CIT, COMB).

Predictors are normalized (Yeo–Johnson transform) and models trained using spatial folds.

### 6.2 Outputs

| Output | Description |
|--------|--------------|
| `trait_maps.tif` | Predicted CWM traits globally |
| `cv_folds.csv` | Mapping of samples to hex and fold |
| `autocorr_summary.csv` | Trait-wise autocorrelation ranges |
| `variogram_plots/` | Visual diagnostics per trait |
| `uncertainty_layers.tif` | Coefficient of variation across folds |

---

## 7. Key Concepts Recap

| Concept | Purpose |
|----------|----------|
| **Variogram** | Quantifies spatial dependence to define independence distance |
| **H3 Hex Grid** | Enforces spatial independence globally |
| **KS-based balancing** | Equalizes trait distribution across folds |
| **Spatial K-Fold CV** | Measures model’s spatial generalization |
| **COV & AOA** | Quantify prediction uncertainty and domain validity |

---

## 8. Scientific Rationale

- **Spatial autocorrelation** violates IID assumptions; ignoring it inflates model performance.  
- **Variograms** provide an empirical measure of ecological scale — how far trait similarity extends.  
- **H3-based grids** operationalize this by grouping spatially dependent points.  
- **Spatial cross-validation** tests transferability across regions and biomes.  
- **KS balancing** avoids bias from uneven trait distribution.

Together, this pipeline ensures model evaluations reflect **true ecological generalization** rather than local interpolation.

---

*Developed for the GEOSENSE project (2025) integrating citizen science, remote sensing, and geostatistics for global trait prediction.*
