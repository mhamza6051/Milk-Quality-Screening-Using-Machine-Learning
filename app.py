import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Milk Quality Screening System",
    page_icon="🥛",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.hero-box {
    background: linear-gradient(135deg, #0f4c81, #3a7ca5);
    padding: 30px;
    border-radius: 20px;
    color: white;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load("models/milk_quality_pipeline.joblib")
    encoder = joblib.load("models/label_encoder.joblib")
    return model, encoder

@st.cache_data
def load_metrics():
    with open("models/model_metrics.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("Milk_Quality_Dataset.csv")

model, label_encoder = load_model()
metrics = load_metrics()
df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Single Prediction", "Batch Prediction", "Model Performance", "Visualizations"]
)

if page == "Home":
    st.markdown("""
    <div class="hero-box">
        <h1>🥛 Milk Quality Screening System</h1>
        <p style="font-size:18px;">
        A realistic ML screening model for milk quality classification using physicochemical features.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Selected Model</h3>
            <h2>{metrics['best_model']}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Test Accuracy</h3>
            <h2>{metrics['test_accuracy']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Features Used</h3>
            <h2>{len(metrics['feature_set_used'])}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Why this model is realistic")
    st.write("""
    This app uses only physicochemical milk properties for classification.
    Direct adulteration indicator variables were excluded to avoid unrealistically inflated accuracy.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

elif page == "Single Prediction":
    st.title("Single Sample Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input("pH", value=6.7, format="%.2f")
        acidity = st.number_input("Titratable Acidity", value=0.16, format="%.3f")
        fat = st.number_input("Fat Percentage", value=3.5, format="%.2f")

    with col2:
        snf = st.number_input("SNF Percentage", value=8.5, format="%.2f")
        lactose = st.number_input("Lactose", value=4.8, format="%.2f")
        protein = st.number_input("Protein", value=3.2, format="%.2f")

    with col3:
        density = st.number_input("Density", value=1.03, format="%.3f")
        conductivity = st.number_input("Conductivity", value=4.5, format="%.2f")
        freezing_point = st.number_input("Freezing Point", value=-0.54, format="%.2f")

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "pH": ph,
            "Titratable_Acidity": acidity,
            "Fat_Percentage": fat,
            "SNF_Percentage": snf,
            "Lactose": lactose,
            "Protein": protein,
            "Density": density,
            "Conductivity": conductivity,
            "Freezing_Point": freezing_point
        }])

        pred = model.predict(input_df)[0]
        label = label_encoder.inverse_transform([pred])[0]

        st.success(f"Predicted Class: {label}")

elif page == "Batch Prediction":
    st.title("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.dataframe(batch_df.head(), use_container_width=True)

        required_columns = [
            "pH",
            "Titratable_Acidity",
            "Fat_Percentage",
            "SNF_Percentage",
            "Lactose",
            "Protein",
            "Density",
            "Conductivity",
            "Freezing_Point"
        ]

        missing_cols = [col for col in required_columns if col not in batch_df.columns]

        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            preds = model.predict(batch_df[required_columns])
            batch_df["Predicted_Quality"] = label_encoder.inverse_transform(preds)

            st.dataframe(batch_df, use_container_width=True)

            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions",
                csv,
                "milk_quality_predictions.csv",
                "text/csv"
            )

elif page == "Model Performance":
    st.title("Model Performance")

    st.metric("Test Accuracy", f"{metrics['test_accuracy']:.2%}")
    st.metric("Test Precision", f"{metrics['test_precision']:.2%}")
    st.metric("Test Recall", f"{metrics['test_recall']:.2%}")
    st.metric("Test F1 Score", f"{metrics['test_f1']:.2%}")

    st.subheader("Cross Validation Results")
    cv_df = pd.DataFrame(metrics["cross_validation_results"])
    st.dataframe(cv_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=cv_df, x="CV Mean Accuracy", y="Model", ax=ax)
    ax.set_title("Cross Validation Accuracy Comparison")
    st.pyplot(fig)

elif page == "Visualizations":
    st.title("Data Visualizations")

    st.subheader("Class Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(data=df, x="Quality_Grade", order=df["Quality_Grade"].value_counts().index, ax=ax)
    ax.tick_params(axis='x', rotation=15)
    st.pyplot(fig)

    st.subheader("Feature Distribution")
    feature = st.selectbox(
        "Select Feature",
        ["pH", "Titratable_Acidity", "Fat_Percentage", "SNF_Percentage",
         "Lactose", "Protein", "Density", "Conductivity", "Freezing_Point"]
    )

    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(data=df, x=feature, hue="Quality_Grade", kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature}")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    corr = df[["pH", "Titratable_Acidity", "Fat_Percentage", "SNF_Percentage",
               "Lactose", "Protein", "Density", "Conductivity", "Freezing_Point"]].corr()
    sns.heatmap(corr, cmap="Blues", ax=ax)
    ax.set_title("Physicochemical Feature Correlation Heatmap")
    st.pyplot(fig)