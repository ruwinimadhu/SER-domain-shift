# SER-domain-shift
Code and data pipeline for managing distribution shift in speech emotion recognition (Random Forest, XGBoost, confidence filtering

# Managing Distribution Shift in Speech Emotion Recognition

This repository contains the code and experiments for the paper:  
**"Managing Distribution Shift and Confounding Variables in Noisy Signal Data: A Case Study in Speech Emotion Recognition with Implications for Biomedical Applications"**

## 📌 Overview
Speech Emotion Recognition (SER) systems often perform well in-domain but degrade sharply under distribution shifts such as:
- **Gender mismatch**
- **Cross-corpus transfer**
- **Low-arousal vs high-arousal emotions**

This work explores the robustness of classical classifiers (Logistic Regression, Random Forests, XGBoost) under these conditions, and introduces a **confidence-based filtering mechanism** to improve reliability on a trusted subset of predictions.

## 🔑 Key Findings
- Random Forest and XGBoost achieve strong in-domain accuracy (>0.85).  
- Cross-domain accuracy collapses (e.g., RF drops to 0.466, XGBoost to ~0.30).  
- Confidence-based filtering recovers accuracy (0.811) at reduced coverage (7%).  
- High-arousal emotions (angry, fear, surprise) generalize better than low-arousal ones (calm, neutral, sad).  

## 🛠️ Repository Structure
├── notebooks/ # Jupyter notebooks for training & evaluation
├── src/ # Python scripts for models, preprocessing, and utils
├── data/ # Links/instructions for datasets (CREMA-D, RAVDESS)
├── results/ # Figures, tables, and metrics
└── README.md


## ⚙️ Requirements
- Python 3.9+
- scikit-learn
- xgboost
- pandas, numpy
- matplotlib / seaborn

Install dependencies:
```bash
pip install -r requirements.txt


🚀 Usage

Download datasets:
CREMA-D - https://github.com/CheyneyComputerScience/CREMA-D?utm_source=chatgpt.com
RAVDESS -10.5281/zenodo.1188975

Preprocess & train:
python src/train_models.py

Evaluate cross-domain performance:
python src/evaluate_cross_domain.py

Run confidence-based filtering:
python src/confidence_filtering.py --threshold 0.6
