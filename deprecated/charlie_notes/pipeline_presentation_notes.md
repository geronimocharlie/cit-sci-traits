

# 🟩 **Slide 1 – Global plant-trait mapping with machine learning**

### 🎙️ **Speaking script**

**Project overview**

* The goal is to create **continuous, spatially explicit maps** of plant functional traits — measurable biological properties that describe how vegetation functions.
* Each map represents the **community-weighted mean (CWM)** of a trait per grid cell, integrating all species present and their relative abundances.
* We model **31 key traits** from the TRY database, covering morphological, physiological, and biochemical dimensions — e.g. specific leaf area (SLA), leaf nitrogen, wood density, plant height, seed mass, etc.

**Scientific scope and technical challenge**

* Spatial resolution ranges from **1 km (fine-scale)** to **~222 km (continental scale)**, enabling both ecosystem- and Earth-system-model integration.
* Data come from multiple sources:

  * **sPlot** – standardized vegetation surveys with species composition and cover,
  * **GBIF** – citizen-science occurrence records,
  * **TRY** – trait measurements,
  * **Earth-observation predictors** (MODIS, WorldClim, SoilGrids, canopy height, vegetation optical depth).
* Machine-learning methods model **nonlinear, multivariate relationships** between environment and traits.
* Ecological data violate the IID assumption due to **spatial autocorrelation** (points close in space are more similar than distant ones).
  → We use **spatial cross-validation** and **range-based splitting** so that training and test sets are geographically independent.

**Deliverables**

* A suite of **global trait maps** at multiple resolutions.
* For each trait:

  * Predicted mean (CWM),
  * **Uncertainty layers** (coefficient of variation across folds),
  * **Area-of-applicability (AOA)** masks showing where predictions are based on similar environmental conditions as the training data.
* Together these provide the first globally consistent, uncertainty-aware plant-trait surfaces.

---

# 🟩 **Slide 2 – Why map plant traits globally?**

### 🎙️ **Speaking script**

**1. Ecological and theoretical foundation**

* Traits are the **fundamental currency** of plant ecology — they determine how plants acquire resources, compete, and tolerate stress.
* Functional traits control key processes:

  * **Leaf-economics spectrum** – trade-off between fast resource acquisition and conservation (SLA, leaf N, photosynthetic rate).
  * **Size and seed spectrum** – trade-off between height, reproduction, and survival.
* The *Global Spectrum of Plant Form and Function* (Díaz et al. 2016) showed that ~75 % of global trait variation lies along these two axes, implying **strong cross-trait correlation** and a low-dimensional trait space.

**2. Relevance for Earth-system science**

* Dynamic Global Vegetation Models (DGVMs) and land-surface models still use **fixed plant-functional-type (PFT) constants**, e.g., “tropical evergreen tree.”
  → This masks real functional diversity and misrepresents fluxes.
* **Continuous global trait fields** allow those models to simulate physiological diversity directly, improving predictions of:

  * carbon assimilation (GPP/NPP),
  * evapotranspiration and water use,
  * vegetation resilience and adaptation to climate stress.



Although one obvious use of these maps is as input layers for vegetation or climate models, they have a much broader value: they enable direct analyses of functional diversity, provide baselines for conservation planning, and act as reference layers for calibrating satellite sensors.
In the Lusk et al. (2025) study, these maps are explicitly positioned as tools to ‘enhance our understanding of plant community properties and ecosystem functioning globally, and to inform conservation efforts.’
They effectively link crowd-sourced biodiversity data to Earth observation, creating a global functional lens on the biosphere — a platform for future monitoring and scientific discovery.



**3. Why machine learning fits**

* Trait correlations mean the system can be learned from environmental and spectral predictors.
* ML can capture nonlinear, context-dependent responses (e.g., temperature × precipitation × soil).
* Multi-target approaches can exploit the **joint covariance among traits**, improving accuracy and ecological realism.

**4. Quantitative impact**

* The current pipeline (Lusk et al. 2025): r ≥ 0.5 for ≈ 15 traits at 1 km.
* Improved models (foundation or multi-trait) could add +0.05–0.10 to r and expand AOA by ≈ 3–7 percentage points.
* Translating to process models: a ~25 % reduction in trait RMSE propagates to **≈ 10 % less uncertainty** in global carbon and water fluxes.
* Hence, methodological advances in ML have direct implications for climate projections and biodiversity modelling.

---

# 🟩 **Slide 3 – Data integration & challenges**

### 🎙️ **Speaking script**

**1. Input data**

| Source                            | Description                                                                                                           | Strength                                 | Limitation                             |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | -------------------------------------- |
| **TRY**                           | Individual-level trait measurements                                                                                   | Accurate, standardized                   | Sparse, taxonomically biased           |
| **sPlot**                         | Vegetation plots with species abundance                                                                               | Quantitative composition, CWM derivation | Limited regional coverage              |
| **GBIF / iNaturalist**            | Citizen-science occurrences                                                                                           | Massive spatial extent, fills gaps       | Sampling bias, unbalanced taxa         |
| **Environmental / EO predictors** | MODIS reflectance (72 bands), WorldClim (6 vars), SoilGrids (61 vars), VODCA optical depth, canopy height, topography | Continuous global coverage               | Different spatial/temporal resolutions |

* Predictors summarize vegetation state, climate, and edaphic gradients at 1 km grid spacing.
* Each observation (point) is a **tabular vector** of hundreds of features — a typical ML table but with strong spatial dependencies.

**2. Outputs**

* For each trait and resolution, the model produces:

  * **Predicted mean (CWM)** values for every grid cell.
  * **Uncertainty layer (Coefficient of Variation, COV)** — derived from variation across CV folds, indicates local model stability.
  * **Area of Applicability (AOA)** — regions in predictor space similar to training data; outside AOA, predictions are marked unreliable.
* **Deliverables** include downloadable GeoTIFF maps and an online interactive viewer (Google Earth Engine).

**3. Data limitations and methodological consequences**

* **Spatial bias:** data dense in Europe, N America, E Asia; sparse in tropics and deserts → requires spatial weighting or AOA filtering.
* **Autocorrelation:** nearby samples share environment → inflate accuracy if untreated → motivates spatial CV and hex-based folds.
* **Trait gaps and taxonomic mismatches:** incomplete linkage between TRY and species occurrences → introduce noise.
* **Predictor heterogeneity:** EO variables measured at different times and sensors; most static snapshots → temporal dynamics not yet captured.

**4. Implication**

* Handling these limitations correctly (via spatial CV, uncertainty quantification, AOA masking) is crucial to producing trait maps that are scientifically valid and usable in global modelling workflows.

---
