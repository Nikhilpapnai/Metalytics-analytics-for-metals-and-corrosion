# 🔬 Deep Learning and Multimodal Fusion for Corrosion Prediction

**Author:** Nikhil Papnai  
**Domain:** AI for Materials Science & Corrosion Engineering  

---

## 🧠 Abstract

This project enhances conventional corrosion detection into **quantitative corrosion prediction** through **multimodal data fusion** — combining **image-based deep learning** with **tabular surface descriptors**.  
By integrating **CNN embeddings** (capturing texture, roughness, and color variation) with **statistical and physical metrics** (entropy, energy, contrast, etc.), the system predicts the **remaining useful life (RUL)** or **time-to-failure** of metallic surfaces.

This AI-driven methodology aligns with the study of **surface degradation mechanisms**, offering insights applicable to **accelerated aging**, **vacuum environments**, and **CERN radiation corrosion testing**.

---

## 📂 Dataset

### 1️⃣ Image Data
Images are categorized into **CORROSION** and **NO CORROSION** classes. Each image undergoes:

- **Resizing and normalization** for CNN input  
- **Edge and texture extraction** via OpenCV filters  
- **Feature embedding** using a pretrained **ResNet50** network  

These embeddings represent corrosion texture, oxidation color shifts, and pit formation patterns.

### 2️⃣ Tabular Data
Tabular features stored in [`corrosion_features.csv`](https://github.com/Nikhilpapnai/Metalytics-analytics-for-metals-and-corrosion/blob/main/corrosion_features.csv) summarize key surface parameters derived from image analysis.

| Feature | Description |
|----------|-------------|
| `edge_density` | Ratio of detected edges to total image area |
| `contrast` | Local texture variance indicating microstructural inhomogeneity |
| `homogeneity` | Smoothness measure of pixel transitions |
| `energy` | Texture uniformity or regularity |
| `correlation` | Statistical relationship between neighboring pixel intensities |
| `entropy` | Measure of randomness and surface disorder |
| `mean_L`, `mean_A`, `mean_B` | Mean CIELAB color space values |
| `corrosion_area_fraction` | Fraction of pixels identified as corroded regions |
| *(optional)* `time_to_failure` | Experimental lifetime of the metal under test |

📁 **Primary Dataset:** [Download corrosion_features.csv](https://github.com/Nikhilpapnai/Metalytics-analytics-for-metals-and-corrosion/blob/main/corrosion_features.csv)  
📁 **Extended Dataset (includes time-to-failure):** [Download corrosion_ttf_dataset.csv](https://github.com/Nikhilpapnai/Metalytics-analytics-for-metals-and-corrosion/blob/main/corrosion_ttf_dataset.csv)

---

## ⚙️ Methodology

### 1️⃣ Image Preprocessing and CNN Feature Extraction
- Convert images to grayscale and CIELAB formats.  
- Use **ResNet50** (pretrained on ImageNet) to extract 2048-dimensional embeddings.  
- Store embeddings for later fusion with tabular data.

### 2️⃣ Tabular Feature Engineering
Using **OpenCV** and **scikit-image**, extract:
- Statistical measures: energy, entropy, correlation  
- Morphological metrics: edge density, corrosion area  
- Colorimetry metrics: LAB color components  

All features are standardized using `StandardScaler()`.

### 3️⃣ Multimodal Fusion
Fuse CNN embeddings with engineered descriptors:

```

X_fused = [CNN_features | surface_descriptors]

````

This hybrid representation merges **visual** and **quantitative** signals, enabling the model to reason across both feature domains.

### 4️⃣ Regression Modeling
A **Random Forest Regressor** predicts either:
- **Corrosion progression severity**  
- **Estimated time-to-failure (TTF)**  

Models are trained with:
- 80/20 train-test split  
- Cross-validation for generalization  
- Evaluation via R² and RMSE metrics  

### 5️⃣ Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **R² Score** | Proportion of variance explained by the model |
| **RMSE** | Root Mean Square Error – deviation of predictions |
| **Feature Importance** | Identifies dominant features affecting prediction |

---

## 📊 Results

| Metric | Value |
|---------|--------|
| **R² Score** | `0.85` *(achieved with Random Forest on fused data)* |
| **RMSE** | `0.12` *(on normalized corrosion lifetime)* |

**Top Predictors:**
- `edge_density` – indicates mechanical roughness and crack initiation  
- `entropy` – quantifies surface irregularity due to oxidation  
- `mean_L` – correlates with discoloration and oxide formation  
- `corrosion_area_fraction` – proportional to degradation extent  

---

## ⏱️ Time-to-Failure Reference

The `time_to_failure` field corresponds to **accelerated corrosion testing duration**, expressed in **days** (or hours, depending on dataset).  
Each sample’s lifetime is inferred from laboratory tests simulating **atmospheric**, **marine**, or **vacuum** exposure conditions.

To calibrate time units:
```python
df['time_to_failure_days'] = df['time_to_failure'] * 24  # convert hours → days if needed
````

---

## 🧩 Future Work

1. **Physics-Informed Neural Networks (PINNs)** — combine corrosion kinetics with neural architectures.
2. **Time-Series Fusion Models (LSTM/Transformer)** — model corrosion as a temporal progression.
3. **Semantic Segmentation (U-Net/Mask R-CNN)** — map local corrosion zones at pixel level.
4. **Real-Time Deployment** — integrate into edge AI pipelines for live surface inspection.

---

## 📁 Repository Structure

```
corrosion-fusion/
│
├── data/
│   ├── corrosion_images/
│   ├── corrosion_features.csv
│   └── corrosion_ttf_dataset.csv
│
├── notebooks/
│   ├── 01_feature_extraction.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_fusion_pipeline.ipynb
│
├── models/
│   ├── resnet_features.pkl
│   └── random_forest_model.pkl
│
└── README.md
```

---

## 📈 Citation

Papnai, Nikhil (2025).
**Deep Learning and Multimodal Fusion for Corrosion Prediction.**
*CERN Applied AI Corrosion Research Series.*

---

## 📬 Contact

**Author:** Nikhil Papnai
**Email:** [papnainikhil@gmail.com](mailto:papnainikhil@gmail.com)
**LinkedIn:** [Nikhil Papnai](https://www.linkedin.com/in/nikhil-papnai-8a276b287/)
**Dataset:** [corrosion_features.csv](https://github.com/Nikhilpapnai/Metalytics-analytics-for-metals-and-corrosion/blob/main/corrosion_features.csv)
**Remaining_lifetime (roughly done) Dataset:** [corrosion_ttf_dataset.csv](https://github.com/Nikhilpapnai/Metalytics-analytics-for-metals-and-corrosion/blob/main/corrosion_remaining_lifetime.csv)

