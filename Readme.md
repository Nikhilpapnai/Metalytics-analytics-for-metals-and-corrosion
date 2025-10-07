# 🔬 Deep Learning and Multimodal Fusion for Corrosion Prediction

**Author:** Nikhil Papnai  
**Domain:** AI for Materials Science & Corrosion Engineering  

---

## 🧠 Abstract

This project extends corrosion detection into **quantitative corrosion prediction** by combining **image-based deep learning** and **tabular surface descriptor analysis**.  
Using both **visual embeddings** from convolutional neural networks (CNNs) and **numeric texture features** (entropy, energy, contrast, etc.), the model estimates the **remaining useful life (RUL)** or **time-to-failure** of corroded metal surfaces.

By fusing image and tabular data, the system links AI-based visual recognition with measurable **surface physics**, enabling a predictive framework aligned with real-world degradation under industrial and CERN-like environments.

---

## 📂 Dataset

### 1️⃣ Image Data
Images of metallic surfaces labeled as **CORROSION** and **NO CORROSION** were collected and categorized into train, validation, and test sets.

Each image is processed through:
- Pixel normalization  
- Edge and texture extraction  
- CNN feature embedding (e.g., ResNet50)

### 2️⃣ Tabular Data
Tabular data (`corrosion_features.csv`) contains image-derived and surface physics metrics.

| Feature | Description |
|----------|--------------|
| `edge_density` | Fraction of detected edges per image area |
| `contrast` | Texture variance representing local intensity fluctuations |
| `homogeneity` | Smoothness measure of surface patterns |
| `energy` | Uniformity in texture distribution |
| `correlation` | Spatial dependency between neighboring pixels |
| `entropy` | Surface randomness and complexity |
| `mean_L`, `mean_A`, `mean_B` | Average CIELAB color components |
| `corrosion_area_fraction` | Area fraction of corroded pixels |
| *(optional)* `time_to_failure` | Degradation timeline in desired time units |

📁 **Dataset Link:** [Download corrosion_features.csv](https://github.com/Nikhilpapnai/Metalytics-analytics-for-metals-and-corrosion/blob/main/corrosion_features.csv)  
👉 *(Replace `#` with your actual file URL)*

---

## ⚙️ Methodology

### 1️⃣ Image Preprocessing and CNN Feature Extraction
- Images are resized, normalized, and passed through a pretrained CNN (ResNet50).
- Extracted embeddings represent high-dimensional corrosion textures and patterns.

### 2️⃣ Tabular Feature Engineering
- Using OpenCV and scikit-image, numerical descriptors are extracted.
- Features like entropy, energy, and color balance quantify corrosion severity.

### 3️⃣ Multimodal Fusion
Visual embeddings and tabular features are concatenated:

```

X_fused = [CNN_features | surface_descriptors]

```

This fusion creates a single model input that integrates both **visual** and **quantitative** corrosion characteristics.

### 4️⃣ Regression Modeling
A **Random Forest Regressor** predicts the **time-to-failure** or degradation severity using fused features.

### 5️⃣ Evaluation Metrics
| Metric | Description |
|---------|-------------|
| R² Score | Goodness of model fit |
| RMSE | Root Mean Square Error (average prediction deviation) |
| Feature Importance | Identifies the most influential parameters |

---

## 📊 Results

| Metric | Value |
|---------|--------|
| **R² Score** | `0.85` *(example value)* |
| **RMSE** | `0.12` *(example value)* |

**Top Predictors:** 
- `edge_density `
- `entropy`  
- `mean_L` (brightness/lightness)  
- `corrosion_area_fraction`  

---

## ⏱️ Time Frame

Specify the scale of the target variable (depends on dataset):

> `time_to_failure` represents degradation timeline measured in  
> **hours**, **days**, or **years** — depending on test setup.

Add or calibrate the field accordingly in your CSV.

---

## 🧩 Future Work

1. **Physics-Informed Neural Networks (PINNs):** Integrate electrochemical models for physics-guided learning.  
2. **Temporal Modeling:** Extend to time-series (LSTM/Transformer) for dynamic corrosion evolution.  
3. **Segmentation Fusion:** Use U-Net/Mask R-CNN for localized corrosion pixel mapping.  
4. **Deployment:** Build a real-time web or mobile inference app for corrosion inspection.

---

## 📁 Repository Structure

```

corrosion-fusion/
│
├── data/
│   ├── corrosion_images/
│   └── corrosion_features.csv
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
**Deep Learning and Multimodal Fusion for Corrosion Prediction**.  
*CERN Applied AI Corrosion Research Series.*

---

## 📬 Contact

**Author:** Nikhil Papnai  
**Email:** *(add email if desired)*  
**LinkedIn:** [Your LinkedIn Profile](https://www.linkedin.com/in/nikhil-papnai-8a276b287/)  
**Dataset:** [corrosion_features.csv](https://github.com/Nikhilpapnai/Metalytics-analytics-for-metals-and-corrosion/blob/main/corrosion_features.csv)

---
```
