# Detailed Results & Experiment Notes

All numbers are taken directly from the experiment notebook
(`notebooks/milk_quality_analysis.ipynb`) and `models/model_metrics.json`.
Split: 80% train (8,000) / 20% test (2,000), stratified. Cross-validation:
5-fold StratifiedKFold, shuffled, `random_state=42`.

## Cross-validation comparison (all 4 models, 9 physicochemical features)

| Model | CV Mean Accuracy | CV Std |
|---|---|---|
| Random Forest | 0.9416 | 0.0044 |
| Gradient Boosting | 0.9405 | 0.0056 |
| SVM | 0.9059 | 0.0099 |
| Logistic Regression | 0.8679 | 0.0083 |

Notebook interpretation (kept from the original): Logistic Regression is lowest
because the class boundaries are nonlinear; SVM improves on it; the ensembles
model the complex relationships best.

## Model selection

Although Random Forest and Gradient Boosting reached ~94% CV accuracy, the
deployed model is **Logistic Regression**, selected (in the original author's
words) because it "provides a more interpretable and generalizable solution"
and its ~87% test performance "reflects a more realistic classification"
for a screening setting. The saved pipeline is StandardScaler + Logistic Regression.

## Deployed model — test-set classification report

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Adulterated | 0.79 | 0.68 | 0.73 | 500 |
| High Quality | 0.91 | 0.95 | 0.93 | 700 |
| Low Quality | 0.83 | 0.90 | 0.87 | 500 |
| Spoiled | 0.98 | 0.95 | 0.96 | 300 |
| **Overall** | | | **Accuracy 0.871** | 2,000 |

Weighted averages: Precision 0.869 · Recall 0.871 · F1 0.869.

## Feature coefficients (deployed Logistic Regression)

| Feature | Coefficient |
|---|---|
| Titratable_Acidity | −1.736 |
| pH | +0.781 |
| Fat_Percentage | +0.684 |
| SNF_Percentage | −0.328 |
| Lactose | −0.047 |
| Protein | −0.035 |
| Density | −0.048 |
| Conductivity | −0.030 |
| Freezing_Point | −0.030 |

(Coefficients shown from the notebook's coefficient analysis of the saved pipeline.)

## Leakage control — the central methodological point

Excluded features: Water_Adulteration_Percent, Urea, Detergent, Starch,
Neutralizers, Preservatives, Antibiotics. In the raw data these separate the
Adulterated class almost perfectly (e.g. mean added-water 22.3% for Adulterated
vs ~2.5% for every other class; urea/detergent/starch present in ~70% of
Adulterated samples and 0% of High/Low Quality). Including them would have
produced a misleading near-perfect score. The reported 87.1% is the honest
screening performance from physicochemical properties alone.
