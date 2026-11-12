# Milk Quality Screening Using Machine Learning

**A 4-class milk quality classifier — High Quality, Low Quality, Adulterated, Spoiled — built on physicochemical properties only, with direct adulteration indicators deliberately excluded to prevent target leakage.**

A milk sample's lab measurements (pH, acidity, fat, SNF, lactose, protein, density, conductivity, freezing point) carry a real signal about its quality. This project trains a screening model on **10,000 milk samples** to grade quality from those measurements alone — the way a realistic screening tool would have to work *before* any adulterant test is run.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-ff4b4b)
![Task](https://img.shields.io/badge/Task-4--Class%20Classification-green)

---

## The key design decision: no leakage

The dataset contains 7 variables that directly reveal adulteration — added water %, urea, detergent, starch, neutralizers, preservatives, antibiotics. A model using them would score near-perfect accuracy by simply reading the answer.

**Those 7 features were deliberately excluded.** The deployed model uses only the 9 intrinsic physicochemical properties, so its **87.1% test accuracy is an honest, realistic screening score** — and its mistakes are informative (adulterated milk that chemically mimics good milk is genuinely hard to catch without the adulterant tests, which is the correct scientific conclusion).

## Results

**Model comparison — 5-fold stratified cross-validation:**

| Model | CV Accuracy |
|---|---|
| Random Forest | 0.9416 ± 0.0044 |
| Gradient Boosting | 0.9405 ± 0.0056 |
| SVM | 0.9059 ± 0.0099 |
| Logistic Regression | 0.8679 ± 0.0083 |

**Deployed model — Logistic Regression** (chosen deliberately over the higher-scoring ensembles for interpretability and generalization; its coefficients can be read as a screening rule):

| Test metric (2,000 held-out samples) | Value |
|---|---|
| Accuracy | **0.871** |
| Precision (weighted) | 0.869 |
| Recall (weighted) | 0.871 |
| F1 (weighted) | 0.869 |

**Per-class F1:** Spoiled **0.96** · High Quality **0.93** · Low Quality **0.87** · Adulterated **0.73** (the hardest class — exactly as expected once the direct adulterant indicators are removed).

**Strongest signals** (LR coefficients): titratable acidity (−1.74), pH (+0.78), fat % (+0.68).

## Dataset

`Milk_Quality_Dataset.csv` — 10,000 samples, 17 columns, no missing values, no duplicates.

| Class | Samples |
|---|---|
| High Quality | 3,500 |
| Adulterated | 2,500 |
| Low Quality | 2,500 |
| Spoiled | 1,500 |

**Features used (9):** pH, Titratable_Acidity, Fat_Percentage, SNF_Percentage, Lactose, Protein, Density, Conductivity, Freezing_Point.
**Excluded on purpose (7):** Water_Adulteration_Percent, Urea, Detergent, Starch, Neutralizers, Preservatives, Antibiotics — direct adulteration indicators that leak the target.

## Method

```
10,000 milk samples (16 features + grade)
        │
        ├─► EDA — class distribution, correlation heatmap
        ├─► Leakage control — drop 7 adulterant-indicator features, keep 9 physicochemical
        ├─► 80/20 stratified split ─► 5-fold CV comparison of 4 models
        ├─► Model selection — Logistic Regression (interpretability over raw CV score)
        └─► Test evaluation — accuracy · per-class report · confusion matrix
        │
        └─► Pipeline + label encoder saved (joblib) ─► Streamlit app
```

## The app

`app.py` — **Milk Quality Screening System** (Streamlit), 5 pages:

- **Home** — selected model, test accuracy, dataset preview
- **Single Prediction** — enter the 9 lab measurements, get the quality grade
- **Batch Prediction** — upload a CSV, download graded results
- **Model Performance** — test metrics + the CV comparison of all 4 models
- **Visualizations** — class distribution, feature distributions by grade, correlation heatmap

### Run it locally

```bash
git clone https://github.com/mhamza6051/Milk-Quality-Screening-Using-Machine-Learning.git
cd Milk-Quality-Screening-Using-Machine-Learning
pip install -r requirements.txt
streamlit run app.py
```

## Repository structure

```
├── app.py                            # Streamlit screening app
├── Milk_Quality_Dataset.csv          # 10,000-sample dataset
├── notebooks/
│   └── milk_quality_analysis.ipynb   # EDA → leakage-aware feature design → CV → evaluation
├── models/
│   ├── milk_quality_pipeline.joblib  # Deployed pipeline (StandardScaler + Logistic Regression)
│   ├── label_encoder.joblib          # Quality-grade label encoder
│   └── model_metrics.json            # Test metrics, CV results, feature sets used/excluded
├── outputs/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   └── confusion_matrix.png
├── docs/
│   └── RESULTS.md                    # Full classification report and experiment notes
├── requirements.txt
└── README.md
```

## Tech stack

Python · scikit-learn · pandas · NumPy · matplotlib · seaborn · joblib · Streamlit

## Limitations & future work

- **Adulterated is the weak class (F1 0.73)** — by design: without adulterant tests, chemically disguised adulteration is only partially visible in physicochemical data. A two-stage system (this screener → targeted adulterant testing for uncertain cases) is the realistic deployment pattern.
- Planned: calibrated probabilities with an "uncertain — send for lab adulterant test" band, and re-checking ensemble models with the same leakage rules for a fair accuracy/interpretability trade-off study.

## Author

**Muhammad Hamza** — Data Scientist / AI-ML Developer
GitHub: [@mhamza6051](https://github.com/mhamza6051)
