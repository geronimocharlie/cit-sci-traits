# Spatial Autocorrelation and Spatial Cross-Validation in Trait Mapping

This document explains the methodology for computing **spatial autocorrelation ranges** and how they are used to define **spatially independent cross-validation folds** in the global trait mapping workflow. It is designed for environmental data scientists working with geospatial trait modeling pipelines (e.g., Lunsk et al., 2025).

---

## 1. Motivation

Ecological and trait data exhibit **spatial autocorrelation** — nearby points are more similar than distant ones due to shared environment and biotic processes.  
Conventional random cross-validation assumes data are **independent and identically distributed (IID)**, which is not true here.  
To prevent over-optimistic model evaluation, the pipeline estimates the **autocorrelation distance** (spatial range of dependence) and uses it to design **spatially aware folds**.

---

## 2. Estimating Spatial Autocorrelation Ranges

### 2.1 Source Data
- Data used: **sPlot vegetation plots** (high-quality and globally distributed).  
- Traits: continuous community-weighted means (CWMs) such as leaf N, SLA, wood density, etc.  
- All missing values are removed per trait.

### 2.2 Spatial Subdivision
- Data are divided by **UTM zone** (for EPSG:4326 coordinates) or other consistent metric projections (e.g., EPSG:6933 for global equal-area grids).  
- Within each zone, distances between coordinates are **true metric distances (in meters)**.

This avoids distortion from geographic coordinates, where distance in degrees is not uniform across latitudes.

### 2.3 Variogram Computation
For each trait within each zone:

1. Compute **empirical variograms**:  
   - For all pairs of points, calculate Euclidean distance \( h \) and semivariance  
     \[ \gamma(h) = \frac{1}{2N(h)} \sum (z_i - z_j)^2 \]
2. Fit a **spherical variogram model** to estimate:  
   - **Range (r):** the distance beyond which spatial dependence vanishes.  
   - **Nugget:** microscale noise or measurement error.  
   - **Sill:** total variance.

This yields one autocorrelation range per trait per zone.

### 2.4 Aggregation Across Zones
Each trait’s autocorrelation range estimates are aggregated across all zones:

- Weighted by the number of samples per zone.  
- Summaries computed: mean, standard deviation, and quantiles (q05, q50, q95).  

The **uncertainty** (std, quantiles) reflects how much the range varies spatially — i.e., heterogeneity in trait autocorrelation across regions.

Example summary table:

| Trait | Mean Range (km) | Std | q05 | q95 |
|:------|----------------:|----:|----:|----:|
| SLA | 946 | 234 | 456 | 1234 |
| Leaf N | 820 | 200 | 430 | 1170 |

---

## 3. Using Ranges to Create Spatial Cross-Validation Folds

### 3.1 Why Spatial Folds?
Standard random CV mixes spatially dependent samples across folds, overestimating model skill.  
Instead, spatial CV groups points into **spatially independent bins** based on the autocorrelation range.

### 3.2 Global Hexagonal Grid (H3)
We use the **H3 geospatial indexing system** to partition the Earth into hexagons.

- Each H3 resolution corresponds to a fixed hex size (edge length in meters).  
- The hex size is chosen based on the **trait’s autocorrelation range** (rounded to the nearest larger H3 scale).  
- Example: a 946 km range → H3 resolution 0 (~1,107 km diameter).

H3 automatically defines hex centroids and boundaries globally — no manual placement needed.

### 3.3 Assigning Observations to Hexagons
For each point:
```python
import h3
hex_id = h3.geo_to_h3(lat, lon, resolution)
```
All points inside the same hex share the same `hex_id`.

The number of hexes is simply the number that contain at least one observation.

### 3.4 Fold Assignment
Folds are created **at the hexagon level**:

1. Each hexagon is assigned to exactly one fold (not split).  
2. Fold assignment is optimized to ensure similar **trait distributions** across folds.  
3. This is achieved by repeated random assignments and selecting the one that maximizes the **Kolmogorov–Smirnov (KS) p-value** — meaning trait distributions across folds are most similar.

Result: folds that are both **spatially independent** and **statistically balanced**.

---

## 4. Coordinate Systems Recap

| CRS | Type | Use Case |
|------|------|-----------|
| EPSG:4326 | Geographic (degrees) | Used only for assigning H3 hexagons |
| UTM (zone-specific) | Projected (meters) | Variogram computation per zone |
| EPSG:6933 | Equal-area global | Used when global raster grid already metric |

Distances are always computed in meters; H3 assignment requires conversion back to latitude/longitude.

---

## 5. Outputs

| Output File | Description |
|--------------|--------------|
| `autocorr_summary.csv` | Mean, std, quantiles of autocorrelation ranges per trait |
| `cv_folds.csv` | Trait, point ID, hex ID, fold ID |
| `variogram_plots/` | Diagnostic plots per trait and zone |

---

## 6. Conceptual Overview

1. **Compute variograms** within zones → estimate autocorrelation ranges.  
2. **Aggregate** across zones to get global mean range and uncertainty.  
3. **Select H3 resolution** so hexes ≈ independence distance.  
4. **Assign hexes to folds** using KS-based balancing.  
5. **Run spatial cross-validation** ensuring realistic generalization assessment.

---

## 7. Key Advantages

- Spatial independence of train/test data.  
- Fold balancing for trait distributions.  
- Automatic, globally consistent grid system.  
- Trait-specific resolution: smaller-range traits → finer grid.  
- Uncertainty quantification from regional variation.

---

### In summary:
> **Variograms define the scale of independence.**  
> **H3 grids operationalize it globally.**  
> **Folds built on hexes enable unbiased spatial cross-validation.**