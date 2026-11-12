# Saved Model Artifacts

| File | Contents |
|---|---|
| `milk_quality_pipeline.joblib` | Deployed pipeline — StandardScaler + Logistic Regression, trained on the 9 physicochemical features |
| `label_encoder.joblib` | LabelEncoder for the 4 grades: Adulterated, High Quality, Low Quality, Spoiled |
| `model_metrics.json` | Test accuracy/precision/recall/F1 (0.871 / 0.869 / 0.871 / 0.869), the 4-model CV table, and the used vs excluded feature lists |

`app.py` loads all three files from this folder. Predict with
`pipeline.predict(df[["pH","Titratable_Acidity","Fat_Percentage","SNF_Percentage","Lactose","Protein","Density","Conductivity","Freezing_Point"]])`
and decode with the label encoder.
