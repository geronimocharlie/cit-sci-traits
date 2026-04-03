# **DETAILED PROPOSAL NOTES (Draft)**

## **Title (working)**

**Towards IID-Compatible Modelling for Non-IID Geospatial Tabular Data: Improving Automated ML Workflows for Environmental Science**

---

# **1. Problem Statement**

### **1.1 The gap in current ML practice**

Most widely used ML methods and AutoML frameworks (XGBoost, LightGBM, CatBoost, Tabular Foundation Models, AutoGluon, AutoSklearn, H2O, PyCaret, TabNet, TabTransformer) assume **IID data**:

- samples are exchangeable
- underlying generating process is constant
- covariates are independent from sample index and position
- splitting strategies (train/val/test) produce representative samples

This is true for tabular data in:

- medical prediction
- credit scoring
- marketing
- industrial QC
- manufacturing
- Kaggle competitions (TabArena, TabRed largely IID)

But **it is almost never true for environmental science**.

### **1.2 In environmental applications, data are fundamentally *non-IID***

Environmental geospatial datasets exhibit:

- **Spatial autocorrelation** (nearby points more similar)
- **Spatially heterogeneous sampling** (clusters from cities, labs, citizen science)
- **Spatial covariate confounding** (climate, soil, land cover vary geographically)
- **Geographical leakage through features** (coords, climate layers, EO products)
- **Non-stationarity** (different processes dominate in different regions)

This violates the assumptions of:

- random cross-validation
- bootstrapping
- standard hyperparameter search
- most feature importance analyses
- marginal uncertainty estimation
- almost all AutoML heuristics

### **1.3 Consequence**

Without intervention:

- **Models report highly inflated accuracy** (random CV)
- **Spatial generalization collapses**
- **Error maps look artificially smooth**
- **Uncertainty is underestimated**
- **Models learn geography, not ecology**
- **Downstream ecological inference is compromised**

This is seen across:

- species distribution modelling
- trait mapping
- carbon stock prediction
- biodiversity metrics
- remote sensing regressions
- yield estimation
- hydrology
- epidemiology

Your project aims to **systematically fix this for tabular ML**.

---

# **2. Case Study: The Plant-Trait Mapping Pipeline**

The “From Smartphones to Satellites” plant-trait mapping paper (Lusk et al. 2025) is the motivating example:

It is a state-of-the-art pipeline combining:

- Citizen science (GBIF)
- Expert vegetation plots (sPlot)
- Trait datasets (TRY)
- Earth observation predictors
- Multiresolution modelling
- Gradient-boosted trees, ensembling, spatial CV

It is one of the first large-scale environmental pipelines evaluating:

- **spatial cross-validation**
- **COV-based uncertainty**
- **AOA domain-of-applicability masks**
- **multi-resolution modelling**

However, despite its quality, the pipeline still suffers from multiple **non-IID failure modes**.

### **2.1 Problems visible in the study pipeline**

### **A. Mixed IID/Non-IID cross-validation strategy**

- Outer folds: **spatial** → correct
- Inner folds: **random** → breaks the assumptions

This creates *data leakage* within the modelling process:

- hyperparameters tuned on IID-like random splits
- models trained under random splits but evaluated under spatial splits
- stacking/ensembling done under IID assumptions
- no guarantee that models optimize for spatial generalization

→ **Mismatch between training signal and evaluation reality.**

---

### **B. Geospatial leakage via predictors**

Environmental predictors (climate, soil, MODIS, VODCA) are:

- highly spatially structured
- correlated with latitude/longitude
- correlated with each other across space

Even without explicit lon/lat features, the model can reverse-engineer location.

→ **Model learns where a pixel is, not what the environment is.**

---

### **C. Trait-space transformations are transductive**

Trait transformations were applied to the **entire dataset before splitting**:

- Yeo–Johnson
- standardization
- multi-trait transformations

This means:

- test distributions influence training normalization
- the pipeline is partially **transductive** (test-time information leaks into training)

Transductive learning is *not wrong*, but:

- must be understood
- its effect on spatial generalization must be examined

---

### **D. Sample weights derived heuristically**

Weights were used to counteract sampling bias in GBIF.

But:

- weighting was not tuned
- not evaluated under alternative schemes
- unclear how weights propagate through stacking or multi-resolution steps
- weights may amplify non-IID biases if they correlate with dense/sparse clusters

---

### **E. No explicit attempt to make the data IID-compatible**

While spatial CV is used, the **predictor space remains non-IID**, causing:

- heavy overfitting in dense regions
- degradation in trait-poor regions
- high COV in deserts, boreal, drylands
- uncertainty maps that reflect sampling, not ecological heterogeneity

---

# **3. Research Goal Based on the Case Study**

The plant-trait mapping pipeline exemplifies a broader class of environmental ML problems:

**Highly non-IID geospatial big data fed into IID-assumption ML tools.**

Your goal is to:

1. **Characterize where and how the pipeline fails due to non-IID structure.**
2. **Explore transformations that make the data effectively IID without losing ecological signal.**
3. **Replace or augment model components with non-IID-robust alternatives.**
4. **Generalize findings toward a systematic framework that AutoML and ML practitioners can apply.**

---

# **4. Three Pathways to “Make the Problem IID”**

These are the **three main axes of your project**.

---

## **PATHWAY 1 — Investigating AutoGluon Stacking as an Implicit Non-IID Correction**

### Hypothesis

AutoGluon’s dynamic stacking might *already* mitigate non-IID structure by acting as an internal distribution-shift corrector.

### Investigation tasks

- Measure performance jumps after each stacking layer
- Compare raw base models vs stacked models under spatial CV
- Track feature/importance drift pre/post stacking
- Measure reduction of spatial autocorrelation in residuals
- Explicitly disable stacking for ablation
- Examine whether stacking transforms predictor distributions toward IID-like patterns

If stacking indeed corrects non-IID structure →

It can inform **how to design explicit non-IID alignment models**.

---

## **PATHWAY 2 — Feature Engineering and Spatial Transformation to Reduce Geospatial Leakage (Make Inputs IID-like)**

### Goal

Remove the *unwanted* geospatial signal (exact location leakage)

while preserving the *desired* ecological signal (environmental context).

### Tasks

### **A. Leakage detection**

- Compute Moran’s I, Geary’s C for each environmental predictor
- Predicting lat/lon from the predictors (location-prediction test)
- Variogram range analysis for each variable
- Mutual information with lon/lat
- Examine whether individual features cluster spatially

### **B. IID-transformations**

- Remove coordinates, or replace them with “distance to ecological regions”
- Residualize environmental variables against smooth spatial surfaces
- Standardize features *within neighbourhoods* instead of globally
- Cluster environmental space (biome/climate clusters) → use cluster IDs instead of raw values

### **C. Adding *context windows* to preserve ecological meaning**

- For each point, derive:
    - local neighbourhood mean
    - local variance
    - local range
    - local gradient
    - k-nearest environmental summaries
    - local PCA
- Window definitions:
    - fixed km radius
    - hex-grid moving windows
    - biogeographical neighbourhoods

This produces **non-location-leaking context features**.

Outcome:

A predictor matrix that is **closer to IID**, but still informative.

---

## **PATHWAY 3 — Switching to Non-IID-Robust Models (TabM)**

### Why TabM?

- Best deep learning tabular model for non-IID conditions (TabArena)
- Outperforms tree models on non-IID (“feature-engineered/industrial”) datasets (TabRed)
- Resilient to distribution shift
- Composable with ensembling
- Works with multi-target modelling

### Tasks

- Replace LightGBM backbone with TabM
- Evaluate spatial generalization vs baseline
- Evaluate COV, AOA, uncertainty decomposition
- Test TabM + context features
- Try multi-target TabM (correlations in traits → better generalization)

Outcome:

A robust modelling backbone that *naturally handles* non-IID data.

---

# **5. From Case Study → General Non-IID Geospatial Framework**

Using the insights from the three pathways, derive a **generalizable set of principles**:

### **Principle 1 — Diagnose spatial non-IID structure**

- leakage tests
- spatial autocorrelation of predictors
- ability to predict coordinates
- spatial residual maps

### **Principle 2 — Use spatial CV correctly**

- outer splits always non-IID
- inner splits must also be non-IID when tuning for spatial generalization
- possibly fewer folds, more repeats

### **Principle 3 — Transform predictor space**

- residualization
- neighbourhood context
- remove proxies for coordinates
- ecological distance encoding
- smoothing unwanted variation

### **Principle 4 — Select non-IID-robust models**

- TabM, TabPFN-family, TabICL
- or ensembles that implement distribution alignment

### **Principle 5 — Re-evaluate uncertainty**

- separate within-AOA and outside-AOA uncertainty
- ensure uncertainty follows ecological heterogeneity, not sampling density

### **Principle 6 — Weighting and balancing**

- density-based weighting
- biome-based weighting
- mixture of sources (expert plots + citizen science)

### **Principle 7 — Create a general “Non-IID Toolkit”**

A modular workflow that environmental ML practitioners can use across tasks:

- species distribution models
- trait mapping
- carbon models
- forest structure
- climate impact assessments
- remote-sensing regressions

---

# **FUTURE RESEARCH DIRECTIONS (clustered thematically)**

---

# **I. Non-IID Detection & Diagnostics**

- Create automatic tests for spatial leakage
- Build a “non-IID index” summarizing:
    - spatial autocorrelation of features
    - model’s ability to predict location
    - instability across random splits
- Design automatic CV strategies based on this index

---

# **II. Alternative Spatial Splitting Strategies**

- Variogram-guided fold size
- Leave-one-biome-out
- Leave-one-continent-out
- Multi-scale CV (1 km → 22 km → 55 km)

---

# **III. Multi-Fidelity / Multi-Resolution Modelling**

- Use coarse-resolution foundation model outputs as features
- Parameter sharing across resolutions
- Multi-resolution ensembles
- Bridging multi-resolution label noise

---

# **IV. Transductive Learning in Ecological ML**

- Clarify when and why the pipeline is transductive
- Evaluate whether trait transformation applied globally leaks future info
- Develop guidelines for safe transductive workflows

---

# **V. Sample Weight Learning & Source Integration**

- Learn sample weights from:
    - density models
    - uncertainty models
    - pairwise consistency
- Unify GBIF + sPlot using principled weighting
- End-to-end weight learning with TabM or TabPFN

---

# **VI. Multi-Target Learning & Trait Embedding Models**

- Joint modelling of correlated traits
- Trait embeddings using TRY database
- Predict trait vectors rather than single traits
- Potentially better for rare / weak traits

---

# **VII. Model Efficiency & Scalability**

- Few folds + more repeats strategies
- Caching to reduce EO preprocessing
- Learning when to skip modelling for low-signal traits

---

# **VIII. Towards a General Framework for Geospatial Tabular ML**

End goal:

- A modelling framework that automatically:
    - identifies non-IID data
    - proposes transformations
    - selects proper CV
    - chooses appropriate models
- Making environmental ML accessible to the broader ML community.

---

## VIII. Modeling trait distributions directly