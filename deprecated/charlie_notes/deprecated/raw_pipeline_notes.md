https://github.com/GeoSense-Freiburg/cit-sci-traits

## 🔹 Spatial aggregation

- **Occurrence data (GBIF, sPlot):**
    - Point-level observations (species presence or plots) are **binned into grid cells** of size 1, 22, 55, 111, or 222 km.
    - Within each grid cell:
        - From **sPlot**: species *relative abundance/cover* is used.
        - From **GBIF**: species occurrences are counted (opportunistic, no abundance).
    - Species in each grid cell are then linked to their traits from **TRY**, and **community-weighted means (CWMs)** are calculated.
- **Trait data (TRY):**
    - Matched to species in GBIF/sPlot before aggregation.
    - Each grid cell thus gets CWMs derived from the trait values available for the species recorded inside it.
- **Environmental predictors (MODIS, WorldClim, SoilGrids, VODCA, Canopy height):**
    - Each predictor is **averaged or resampled** to the exact same modeling grid (1–222 km).
    - Example:
        - MODIS reflectance (native 500 m–1 km) → averaged over all pixels within each modeling cell.
        - SoilGrids (native 250 m) → aggregated by mean/median within each grid cell.
        - VODCA (native 25 km) → downscaled/upscaled to align with 1–222 km grids.
        - WorldClim (native ~1 km) → aggregated to coarser grids.
- **Key detail:**
    - Instead of producing a **single high-res product and then upscaling**, the study **trains independent models at each resolution**.
    - This avoids scale-transfer bias (e.g. smoothing artifacts from naive upscaling).

---

## 🔹 Temporal handling

- **Occurrence & trait data (GBIF, sPlot, TRY):**
    - These are **treated as temporally static**.
    - Even though GBIF spans 1700s–present and TRY spans decades, the models do not incorporate time explicitly.
    - Each record contributes to species distribution/traits regardless of sampling year.
- **Environmental predictors:**
    - MODIS is daily, VODCA spans 1987–2016, WorldClim is 1950–2000 normals, SoilGrids is essentially static.
    - In this study, **all predictors are used as long-term climatologies or temporally aggregated “static layers.”**
    - Example:
        - MODIS reflectance: instead of using seasonal/daily time series, it is averaged/aggregated into a static mean value per grid cell.
        - VODCA: multiyear mean vegetation optical depth used.
        - WorldClim: fixed climatology (no interannual variation).
- **Implication:**
    - The model produces **static trait maps** (not time-varying).
    - The authors note that incorporating **temporal dynamics** (e.g., using MODIS seasonality or year-to-year shifts) is a promising next step, but trait measurements are too sparse in time to support that yet.

---

✅ **In summary:**

- Spatial aggregation = resampling all data (occurrences, traits, environment) into common modeling grids at multiple scales (1–222 km).
- Temporal scales = collapsed to **long-term averages** or treated as **static**, since trait/occurrence data are temporally sparse and uneven.

## 🔹 Inputs to the Model

1. **Reference data (the “training labels”):**
    - **Community-weighted mean (CWM) traits** per grid cell, derived by combining:
        - **Species occurrences** (from GBIF or sPlot, or both).
        - **Trait values** (from TRY).
    - Example: In a 22 km grid cell, if species A, B, C occur, and their traits are known from TRY, the model computes the average trait value weighted by relative abundance (sPlot) or occurrence frequency (GBIF).
    - This gives one **numeric target value per trait per grid cell** (e.g. SLA = 14 m²/kg for that cell).
2. **Predictor variables (the “features”):**
    - **Remote sensing & environmental layers** aggregated to the same resolution:
        - MODIS reflectance (optical, 500 m–1 km → resampled).
        - WorldClim bioclimatic variables (~1 km).
        - SoilGrids (250 m).
        - VODCA vegetation optical depth (~25 km).
        - Global canopy height (~1 km).
    - Each grid cell is thus described by ~150 predictor values (climate, soil, reflectance, structure).

---

## 🔹 The Model

- **Type:** Gradient-Boosted Decision Trees (GBDTs).
    - Chosen for ability to capture non-linear relationships and handle mixed predictors.
- **Setup:**
    - Separate models are trained for each **trait (31 traits)** and each **resolution (1, 22, 55, 111, 222 km)**.
    - Data subsets:
        - **SCI** (sPlot vegetation surveys only).
        - **CIT** (citizen science / GBIF only).
        - **COMB** (combined, with down-weighting of noisy citizen science).
- **Training procedure:**
    - Models are trained with spatial cross-validation (holding out regions to test spatial generalization).
    - Predictors are Yeo–Johnson transformed for normalization.

---

## 🔹 Outputs

1. **Trait maps (main output):**
    - Global maps of **community-weighted mean (CWM) values** for 31 plant traits.
    - At multiple resolutions (1–222 km).
    - Delivered as **GeoTIFF rasters** (usable in GIS / Earth Engine).
2. **Uncertainty layers:**
    - **Coefficient of Variation (COV):** shows where model predictions are less stable.
    - **Area of Applicability (AOA):** mask of where predictor conditions fall outside the training domain (i.e., extrapolation).
3. **Evaluation results:**
    - Correlation (r) and error metrics against **independent vegetation survey CWMs** not used in training.
    - Performance varies by trait, but many achieve r ≥ 0.5 at 1 km.

---

## 🔹 Evaluation Pipeline

### 1. **Model Training Setup**

- For each trait (31) × resolution (1, 22, 55, 111, 222 km) × data subset (SCI, CIT, COMB), a **separate GBDT model** is trained.
- Predictors: ~150 environmental variables (static, always the same).
- Targets: community-weighted mean (CWM) traits, depending on SCI/CIT/COMB.

---

### 2. **Cross-validation strategy (splits)**

- They use **spatial cross-validation**, not random splits.
    - Training and test data are split by **geographic blocks**, so test cells are spatially independent of training cells.
    - This avoids overestimating performance due to spatial autocorrelation.
- Evaluation metric: correlation coefficient (r) + normalized RMSE (nRMSE).

---

### 3. **Validation Data**

- Independent test set comes from **held-out sPlot vegetation surveys** that were not used in training.
- This ensures a consistent, systematic benchmark across SCI, CIT, and COMB models.

---

## 🔹 Dealing with Heterogeneity & Bias

- **Down-weighting noisy samples:**
    - In COMB models, citizen science observations are explicitly down-weighted so expert survey data dominate signal quality.
- **Multiple subsets:**
    - By training SCI, CIT, and COMB separately, they can evaluate how each data type contributes (coverage vs. reliability).
- **Independent validation:**
    - Using only expert survey data for testing avoids “self-validation” with noisy citizen science inputs.

---

## 🔹 Non-IID Problem (non-independent, non-identically distributed data)

- **Why it matters:**
    - Data are heavily biased: Europe, North America, Japan, Australia are overrepresented; tropics, deserts, alpine regions underrepresented.
    - If models were validated with random splits, performance would look artificially high (since train/test would share similar regions).
- **Mitigation strategies:**
    - **Spatial cross-validation** ensures models must predict across different regions (closer to real-world deployment).
    - **Area of Applicability (AOA):**
        - Identifies where predictions are extrapolations beyond the training domain (helps flag unreliable outputs).
    - **Coefficient of Variation (COV):**
        - Quantifies uncertainty across folds in cross-validation; high COV = model unstable due to limited training analogs.
    - **COMB models** improve generalization by merging structured surveys with citizen science, broadening coverage.

## 🔹 Why not random CV?

- Random CV would let training and test samples fall very close together (sometimes in the same region).
- Because ecological and remote sensing data are **spatially autocorrelated**, random splits would **inflate performance metrics**.

---

## 🔹 Geographic Block Cross-validation Design

- **Unit of split:** Grid cells at the chosen modeling resolution (1, 22, 55, 111, or 222 km).
- **Blocking:**
    - Grid cells are grouped into **spatially contiguous geographic blocks**.
    - A block = cluster of neighboring cells, so entire regions are held out at once.
- **Folds:**
    - Models are trained on some blocks and tested on others.
    - This ensures test data are **geographically independent** of training data.
- **Implementation:**
    - They use a **systematic tiling** of the global grid (not arbitrary splits).
    - Each fold corresponds to different geographic subsets, covering a range of environments.
    - In practice, it resembles “checkerboard” partitioning — alternating blocks assigned to training vs. test.

---

## 🔹 Independent Validation Data

- The **held-out test set** consists of vegetation survey (sPlot) CWMs that were **not used in training**.
- This is key:
    - Even if CIT data are noisy, evaluation is always against **systematic expert survey plots**.
    - Provides a stable, unbiased benchmark.

---

## 🔹 Benefits of Geographic Block CV

- Prevents **spatial leakage** (nearby training/test points artificially boosting performance).
- Tests **spatial transferability**: can the model trained in one region generalize to another?
- Reveals where predictions fail due to **regional biases** (e.g. Portugal: dense citizen science, few surveys).
- Produces more realistic performance estimates for global mapping tasks.

---

## 🔹 Cross-validation scheme

- **Purpose:** Prevent spatial autocorrelation from inflating performance and assess **spatial transferability** of trait models.
- **Method:** **Spatial K-fold cross-validation**.
- **Targets:** Community-weighted mean (CWM) trait values from **sPlot surveys**.
- **Process:**
    1. The study area is divided into **K spatial folds**, each consisting of one or more **contiguous geographic blocks of grid cells** (size depends on resolution).
    2. In each round:
        - **Training set:** all folds except one.
        - **Test set:** the held-out fold.
    3. This is repeated until **each fold has served once as the test set**.
- **Rotation logic:**
    - If K = 5 folds:
        - Round 1: Train on folds 2–5, test on fold 1.
        - Round 2: Train on folds 1, 3–5, test on fold 2.
        - … until fold 5 has also been tested.
    - Ensures that every region contributes both to training and to independent testing.
- **Evaluation metrics:** Correlation (r), normalized RMSE (nRMSE).
- **Diagnostics added after CV:**
    - **Coefficient of Variation (COV):** For each pixel, compute variation across its predictions in different folds (σ/μ).
    - **Area of Applicability (AOA):** For each pixel, check if its predictor values fall inside or outside the training domain.

---

## 🔹 How geographic blocks are formed

- **Base units:** Grid cells at the chosen modeling resolution (1, 22, 55, 111, 222 km).
- **Grouping into folds:**
    - Cells are grouped into **spatially contiguous blocks**, either by **checkerboard tiling** or **spatial clustering (e.g., k-means on geographic coordinates)**.
    - Each block contains many neighboring cells to enforce spatial independence.
- **Block sizes:**
    - At **1 km resolution:** folds are composed of many small cells, forming blocks that cover areas ~tens to hundreds of km across.
    - At **111–222 km resolution:** each fold can be as large as a **country or sub-continental region** (e.g., Spain, Madagascar, Eastern US).
- **No biome stratification:** Folds are purely geographic, not matched to ecological regions.
- **Implication:** Some test folds contain environments or biomes missing in training — by design, to test true **out-of-region generalization**.

---

✅ **In one line:**

The study uses **K-fold spatial cross-validation with rotating held-out geographic blocks**, so that each region of the world is once withheld for testing while the rest is used for training, ensuring evaluation reflects global spatial generalization rather than local autocorrelation.

# Evaluation - Assesment of Uncertainty

---

## 🔹 What AOA is in general

- **AOA = Area of Applicability.**
- It quantifies whether an **inference pixel** has predictor values similar enough to those used in **model training**.
- Implemented via the **Dissimilarity Index (DI)** method (Meyer & Pebesma 2021).

---

## 🔹 How it is computed in this study

1. **Training data preparation**
    - Each grid cell used in training has a vector of **predictor values** (MODIS reflectance, WorldClim, SoilGrids, VODCA, canopy height).
    - This forms the **reference set of conditions** seen by the model.
2. **Dissimilarity Index (DI)**
    - For every new grid cell at inference, compute the **distance in predictor space** between its vector and the vectors in the training set.
    - Typically, **Euclidean or Mahalanobis distance** in standardized predictor space is used.
    - The **minimum distance** to training points is taken as that cell’s DI.
3. **Thresholding**
    - During cross-validation, the distribution of DI values is compared to prediction error.
    - A **cutoff threshold** is chosen:
        - If DI is **below threshold** → pixel is *inside AOA* (in-domain).
        - If DI is **above threshold** → pixel is *outside AOA* (out-of-distribution).
4. **Output values**
    - AOA is reported as a **binary mask** (inside vs. outside).
    - Additionally, a **continuous dissimilarity map** is available, showing *relative similarity* (lower DI = higher AOA).
    - In results, they summarize AOA as the **percentage of global pixels** that fall inside the AOA for each model.

---

## 🔹 Values and Interpretation in this study

- **With SCI-only training (vegetation surveys):**
    - AOA is smaller, because surveys are clustered in Europe, North America, etc.
    - Many environments (tropics, deserts, alpine) fall outside AOA → model extrapolates there.
- **With COMB (SCI + CIT):**
    - AOA increases for *all traits at all resolutions*.
    - Average global gain = **+2.4 percentage points**.
    - Max gain = **+9.2 points** (wood fiber length at 1 km).
    - Effect is strongest at **finer resolutions (1 km)**, because CIT data add many fine-scale observations.
- **Interpretation of AOA maps:**
    - **High AOA (low DI)** = reliable prediction (training domain covered).
    - **Low AOA (high DI)** = extrapolation (training domain not covered).
    - Magenta areas in maps = outside AOA for both SCI & COMB (e.g. deserts, alpine).
    - Olive areas = unreliable in SCI but reliable in COMB (CIT expanded coverage).

---

## 🔹 Why CIT increases AOA

- **CIT data are not predictors** — the predictor stack (MODIS, climate, soils, etc.) is always the same.
- What CIT does is add **many more training cells** (species + traits matched to TRY) across a wider range of environments.
- This expands the **envelope of training predictor values**.
- Result: during inference, more pixels find “similar” training conditions → more pixels fall inside AOA.

---

✅ **In summary:**

- AOA in this study is computed with a **dissimilarity index in predictor space**, thresholded against training conditions.
- Values are provided as both **continuous dissimilarity maps** and **binary masks**.
- **High AOA = interpolation, reliable. Low AOA = extrapolation, unreliable.**
- Adding CIT increases AOA because it broadens the environmental domain of training data, not because it changes predictors.

---

Yes — let’s go step by step and be precise about how **COV (Coefficient of Variation)** is implemented in this study:🔹 What COV is

---

- COV measures **relative variability** of predictions across cross-validation folds.
- Formula:

COV=σμ\text{COV} = \frac{\sigma}{\mu}

where:

- σ\sigma = standard deviation of predictions across CV folds
- μ\mu = mean prediction across CV folds

---

## 🔹 How it’s computed in this study

1. **Spatial cross-validation setup**
    - The world is split into **geographic blocks** (depending on resolution).
    - For each fold: train on some blocks, predict on held-out blocks.
2. **Prediction collection**
    - Each grid cell is predicted multiple times (once per fold in which it’s held out).
    - This gives a set of predicted trait values for that pixel.
3. **Variation calculation**
    - Compute the **mean prediction** (μ\mu) for the pixel across folds.
    - Compute the **standard deviation** (σ\sigma) across those same predictions.
    - Calculate COV = σ/μ\sigma / \mu.
4. **Output maps**
    - Pixel-wise COV maps are produced for each trait and resolution.
    - Lower COV = model is consistent across folds → stable prediction.
    - Higher COV = predictions vary depending on which blocks were in training → uncertainty.

---

## 🔹 Interpretation in this study

- **Low COV:**
    - Predictions are stable regardless of training subset.
    - Indicates strong, generalizable relationships between traits and predictors in that environment.
- **High COV:**
    - Predictions fluctuate depending on training folds.
    - Indicates **uncertainty due to sparse or biased training data** in that region.
    - Common in deserts, alpine regions, tropics with few surveys.
- **Effect of COMB (SCI + CIT):**
    - COMB models consistently show **lower COV** than SCI-only models.
    - Because adding CIT data reduces data sparsity and increases geographic/environmental coverage.

---

# Piepeline Impelementation

Raw → Interim → Features → Processed → Models → Results.

‘Data’ 

raw/gbif/*.dvc
-> Load raw GBIF data (parquet)
-> Filter by plant functional type (PFT) using config (e.g., only Shrub/Tree/Grass)
-> Set index to species name
-> Subsampling (if enabled): randomly select a fraction of observations (e.g., 1%) to reduce data size for testing or balancing
-> Join with trait data (see below)
-> Reproject coordinates:
- If CRS is not EPSG:4326, convert to EPSG:6933 (Equal Area)
- Uses longitude/latitude to projected x/y
-> Assign grid cell (row, col) using spatial transform:
- Grid resolution from config (e.g., 1km, 55km)
- Each observation mapped to a grid cell based on x/y
-> Masking/filtering:
- Remove observations outside study area (using mask raster or bounding box)
- Filter by minimum/maximum count per cell (e.g., min_count=10, max_count=500)
-> Aggregate observations per grid cell:
- For each cell, compute mean trait values, count of observations, etc.
- If more than max_count, randomly subsample to max_count
-> Output: gridded GBIF data with trait info

raw/TRY_*.dvc, try_pft_*.csv.dvc, TRY_trait_table_*.txt.dvc
-> Load raw trait tables (parquet, csv, txt)
-> Filter for relevant traits (from config)
-> (Optional) Perform PCA on trait matrix if configured (reduces dimensionality)
-> Map species to PFTs (using lookup table)
-> Aggregate trait values per species:
- Compute mean, median, or other summary statistics for each trait
- Remove species with missing or insufficient data
-> Output: trait table ready for joining with GBIF

raw/esa_worldcover_*.dvc, ETH_GlobalCanopyHeight_*.dvc, modis_sur_refl_*.dvc, soilgrids_*.dvc, vodca_*.dvc, wc2-1_30s_bio.dvc
-> Load raster EO datasets
-> Reproject to target CRS (e.g., EPSG:6933)
-> Resample to target grid resolution (e.g., 1km, 55km) using nearest neighbor or bilinear interpolation
-> Masking:
- Apply study area mask (e.g., using biomes.tif or bounding box)
- Set values outside mask to NaN or ignore
-> Stack features:
- For each grid cell, extract EO values (e.g., canopy height, soil, climate)
- Optionally aggregate (mean, min, max) if multiple rasters per cell
-> Output: EO feature grid (per cell)

raw/splot4-0.dvc, sPlotOpen_v76.dvc
-> Load plot-level data
-> Filter for relevant plots/species (e.g., by region, PFT)
-> Map to grid cells (using coordinates and grid transform)
-> Aggregate statistics per cell:
- Compute mean, count, or other summary stats for each plot variable
-> Masking:
- Remove plots outside study area
-> Output: gridded plot data

(GBIF + TRY traits + EO data + SPlot)
-> Join GBIF observations with trait data (by species)
-> Assign each observation to a grid cell (spatial binning, e.g., 1km or 55km)
-> For each grid cell:
-> Aggregate trait values (mean, count, etc.)
-> Aggregate EO features (mean, min, max, etc.)
-> Aggregate plot-level data if available
-> Apply mask to exclude cells outside study area
-> Output: Feature matrix per grid cell (ready for modeling)

(GBIF + TRY traits)
-> For each trait or FD metric:
-> Aggregate trait values per grid cell (mean, count, etc.)
-> If FD metric: calculate functional diversity stats (e.g., f_ric, f_eve) using trait values and species composition
-> Rasterize aggregated values to GeoTIFF (using grid resolution from config)
-> Apply mask to exclude cells outside study area
-> Output: Target raster maps (GeoTIFFs) for each trait/metric

---

---

---

---

**GBIF Observations** (original: point data, lat/lon, ~meters, presence-only)

|

v

**TRY Trait Data** (original: species-level table, global, no spatial resolution)

|

v

+-------------------------------------------------------------+

| 1. Match GBIF observations to TRY traits                    |

|    - By species name                                         |

|    - Each GBIF record gets species-level trait values        |

|    - No abundance info → treat all records equally           |

+-------------------------------------------------------------+

| **Note:** This yields *frequency-weighted means* when       |

| aggregated (not true CWMs).                                 |

+-------------------------------------------------------------+

|

v

+-------------------------------------------------------------+

| 2. Aggregate GBIF+TRY to grid                               |

|    - Reproject coordinates if needed (EPSG:4326 → EPSG:6933)|

|    - Map to target grids (1, 22, 55, 111, 222 km)           |

|    - For each grid cell:                                    |

|        - Mean trait value across all occurrences             |

|        - Observation count                                   |

|    - Always aggregate **directly from raw records** for      |

|      each target grid (not upscaled from 1 km)               |

+-------------------------------------------------------------+

---

**sPlot Vegetation Plot Data** (original: plots, ~10–400 m², up to 1 ha)

|

v

**TRY Trait Data** (species-level, global)

|

v

+-------------------------------------------------------------+

| 3. Compute Community-Weighted Means (CWM) at plot level     |

|    - Match species in plot to TRY traits                     |

|    - Weight by species *relative cover/abundance*            |

|    - Sum over species → one CWM per plot                     |

+-------------------------------------------------------------+

|

v

+-------------------------------------------------------------+

| 4. Aggregate sPlot CWMs to grid                             |

|    - Reproject coordinates if needed                         |

|    - Map plots to target grids (1, 22, 55, 111, 222 km)      |

|    - For each grid cell:                                    |

|        - Mean of plot-level CWMs                            |

|        - Count of contributing plots                        |

|    - Always aggregate **directly from plots** at each target |

|      resolution (not from 1 km grids)                       |

+-------------------------------------------------------------+

---

**Earth Observation (EO) Data** (original: rasters, MODIS 500m–1 km, SoilGrids 250m, WorldClim 1 km, VODCA 25 km, canopy height 1 km)

|

v

+-------------------------------------------------------------+

| 5. Resample EO data to target grid                          |

|    - Reproject to EPSG:6933                                  |

|    - Aggregate to match target grid (1, 22, 55, 111, 222 km)|

|    - For each grid cell:                                    |

|        - Mean, min, max, etc. of predictor variables         |

|    - Creates consistent predictor stack                     |

+-------------------------------------------------------------+

---

+-------------------------------------------------------------+

| 6. Align all sources on common grids                        |

|    - GBIF+TRY (frequency means), sPlot+TRY (CWMs), EO data  |

|    - Aligned to EASE-Grid v2.0 (EPSG:6933)                  |

+-------------------------------------------------------------+

+-------------------------------------------------------------+

| 7. Final Model Inputs & Outputs                             |

|    **Inputs:** EO predictor stack (~150 vars)               |

|    **Targets:** Trait CWMs per grid (SCI, CIT, COMB)        |

|    **Outputs:**                                             |

|       - Global trait maps (GeoTIFFs, 1–222 km)              |

|       - COV maps (uncertainty)                              |

|       - AOA masks (applicability)                          |

+-------------------------------------------------------------+

## Key contrast with sPlot

- **sPlot:** community-weighted mean (CWM) at plot level, weighted by *relative cover/abundance* (continuous measure).
- **GBIF:** frequency-weighted mean at grid level, weighted by *number of occurrence records* (discrete counts, often biased by sampling effort).

---

**Functional diversity (FD) metrics** are quantitative measures that describe the diversity of functional traits within a community of organisms (e.g., plants in a grid cell). Instead of just counting species, FD metrics consider how different the species are in terms of their traits (such as leaf size, height, etc.). Common FD metrics include:

- **Functional Richness (f_ric):** Measures the range of trait values present; how much "trait space" is filled.
- **Functional Evenness (f_eve):** Assesses how evenly trait values are distributed among species.
- **Functional Divergence (f_div):** Indicates how far trait values are from the center of trait space.
- **Functional Redundancy (f_red):** Quantifies overlap in trait values among species.
- **Species Richness (sp_ric):** Simple count of species (not strictly FD, but often included).
- **Functional Richness SES (f_ric_ses):** Standardized effect size for functional richness.

**Why randomly subsample to max_count?** If a grid cell has more than max_count observations, we randomly select max_count of them to use for aggregation and FD calculation. This is done to:

- **Avoid bias:** Prevent cells with many observations from dominating analyses.
- **Ensure comparability:** Make results across grid cells more comparable, since each cell uses a similar number of samples.
- **Reduce computation:** Large numbers of observations can slow down calculations and may not add much new information.
- **Prevent overrepresentation:** Random subsampling ensures that no single cell’s diversity is artificially inflated due to sampling effort.

### Key Details

- **Subsampling**: Randomly selects a fraction or a fixed number of observations to reduce data size or balance samples, especially when there are too many records per cell.
- **Aggregation**: For each grid cell, computes summary statistics (mean, count, diversity metrics) from all observations in that cell.
- **Resolutions**: Grid cell size is configurable (e.g., 1km, 55km, etc.), affecting how fine or coarse the spatial aggregation is.
- **Masking/Filtering**: Removes data outside the study area using raster masks or bounding boxes; filters out grid cells with too few or too many observations.
- **Reprojection**: Converts geographic coordinates to projected coordinates for accurate spatial binning and rasterization.

—> additional outputs to mean trait values of the models

---

### Scripts in ‘Data’

- `__init__.py`: Marks the folder as a Python package; usually empty or contains package-level imports.
- `back_transform_predict.py`: Converts model predictions from transformed (e.g., log, scaled) space back to original units.
- `build_final_metadata.py`: Compiles and writes metadata for final data products (e.g., trait maps, rasters).
- `build_final_product.py`: Assembles the final output datasets or maps from processed/interim results.
- build_gbif_maps.py: Matches GBIF observations with trait data, grids them, computes statistics, and exports trait/FD maps as rasters.
- `build_splot_maps.py`: Processes SPlot plot-level data, grids it, computes statistics, and exports trait maps as rasters.
- `build_try_traits.py`: Loads, filters, and processes TRY trait data, aggregates by species/PFT, and prepares trait tables.
- `consolidate_biomes.py`: Merges or harmonizes biome data from different sources into a unified format.
- `extract_splot.py`: Extracts and preprocesses SPlot data for use in mapping and analysis.
- `harmonize_eo_data.py`: Standardizes and aligns Earth Observation datasets (e.g., reproject, resample, mask).
- `mask.py`: Applies spatial masks to datasets to restrict analysis to the study area.
- `match_gbif_pfts.py`: Matches GBIF species observations to plant functional types (PFTs) using lookup tables.
- `push_sftp.py`: Uploads processed data products to a remote server via SFTP.
- `reproject_final_maps_for_web.py`: Reprojects final raster maps to web-friendly coordinate systems for visualization.
- `standardize_other_products.py`: Harmonizes and standardizes external trait or EO products for integration.
- `subsample_gbif.py`: Randomly subsamples GBIF data to reduce size or balance samples.
- `xfer_gs_assets_to_gee.py`: Transfers GeoSense assets to Google Earth Engine for further analysis or sharing.