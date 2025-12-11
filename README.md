# 🎗️ OncoGuard: Intelligent Breast Cancer Screening System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oncoguard-breast-cancer-ai-ihkfmsvqx3amzkewpfuomj.streamlit.app/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://www.tensorflow.org/)
[![Precision Medicine](https://img.shields.io/badge/Medical_AI-Diagnostic-blue.svg)](https://pubmed.ncbi.nlm.nih.gov/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow.svg)](https://www.python.org/)

## 📄 Abstract
Breast cancer is the most common cancer worldwide. Early diagnosis significantly improves survival rates. However, manual interpretation of Fine Needle Aspiration (FNA) cytology is time-consuming and prone to human variability.

**OncoGuard AI** is a **Clinical Decision Support System (CDSS)** that utilizes a **Deep Neural Network** to analyze 30 cytological features of breast masses. Unlike standard models, this system is engineered to handle **Class Imbalance** (using computed class weights) to prioritize sensitivity, ensuring high accuracy in detecting malignant tumors.

> **[🔴 Launch Live Diagnostic Tool](https://oncoguard-breast-cancer-ai-ihkfmsvqx3amzkewpfuomj.streamlit.app/)**

## 💡 Key Engineering Features
This project implements advanced Data Science techniques to ensure clinical reliability:

* **⚖️ Handling Imbalanced Data:** The dataset contained 63% Benign and 37% Malignant cases. A standard model would be biased towards "Benign." I implemented **Computed Class Weights** to penalize False Negatives heavily, forcing the model to learn Malignant features aggressively.
* **📊 Cytological Profiling (Radar Charts):** The app visualizes the patient's cell features (Radius, Texture, Smoothness) against the **Population Average**, giving doctors an instant visual cue of abnormality.
* **🧠 Robust Architecture:** The Neural Network utilizes **Dropout (30%)** and **Batch Normalization** to prevent overfitting, ensuring the model generalizes well to new patients.

## 🛠️ Technical Architecture
* **Algorithm:** Multi-Layer Perceptron (MLP) built with TensorFlow/Keras.
* **Input Layer:** 30 Features (Standardized using `StandardScaler`).
* **Hidden Layers:**
    * Dense (64 Neurons) + Batch Normalization + Dropout (0.3)
    * Dense (32 Neurons) + Batch Normalization + Dropout (0.2)
* **Output:** Sigmoid Activation (Probability 0-1).
* **Optimizer:** Adam (Adaptive Moment Estimation).

## 📊 Dataset Details
The model was trained on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.
* **Sample Size:** 569 patients.
* **Features:** 30 real-valued inputs computed from digitized images of FNA mass.
    * *Examples:* Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, etc.
* **Target:** Malignant (M) or Benign (B).
* **Performance:** ~96.5% Validation Accuracy.

## 💻 Installation & Usage

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repository
git clone [https://github.com/Muhammad-Shahan/OncoGuard-Breast-Cancer-AI.git](https://github.com/Muhammad-Shahan/OncoGuard-Breast-Cancer-AI.git)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app.py
