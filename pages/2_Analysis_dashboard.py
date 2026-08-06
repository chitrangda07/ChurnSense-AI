"""
pages/2_Analysis_dashboard.py

Analytics Dashboard
"""
import joblib
import pandas as pd
import streamlit as st

from utils.plots import (
    plot_kpi_cards,
    plot_churn_distribution,
    plot_contract_analysis,
    plot_distributions,
    plot_boxplots,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Analytics Dashboard",
    layout="wide"
)

# ---------------------------------------------------------
# Load CSS
# ---------------------------------------------------------

with open("assets/style.css") as css:
    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("Analytics Dashboard")

st.write(
"""
Explore customer behaviour, feature relationships,
and model performance through interactive visualizations.
"""
)

st.divider()

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

@st.cache_data
def load_dataset():

    return pd.read_csv(
        "data/processed/processed_data.csv"
    )

df = load_dataset()

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    return joblib.load(
        "models/xgboost_best.pkl"
    )

model = load_model()

# ---------------------------------------------------------
# Load Metrics
# ---------------------------------------------------------

# ---------------------------------------------------------
# Load Model Metrics
# ---------------------------------------------------------

metrics_df = pd.read_csv(
    "results/model_comparison.csv"
)

xgb_metrics = metrics_df[
    metrics_df["Model"] == "XGBoost"
].iloc[0]

accuracy = float(xgb_metrics["Accuracy"])

roc_auc = float(xgb_metrics["ROC AUC"])

f1_score = float(xgb_metrics["F1 Score"])

# ---------------------------------------------------------
# Load Prediction Results
# ---------------------------------------------------------

y_test = (
    pd.read_csv("results/y_test.csv")
    .squeeze()
)

y_pred = (
    pd.read_csv("results/y_pred.csv")
    .squeeze()
)

y_prob = (
    pd.read_csv("results/y_prob.csv")
    .squeeze()
)

# ---------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------

plot_kpi_cards(
    df=df,
    accuracy=accuracy,
    roc_auc=roc_auc,
    f1_score=f1_score
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Customer Overview
# ---------------------------------------------------------

st.subheader("Customer Overview")

col1, col2 = st.columns(2)

with col1:

    plot_churn_distribution(df)

with col2:

    plot_contract_analysis(df)

st.divider()

# ---------------------------------------------------------
# Customer Behaviour
# ---------------------------------------------------------

st.subheader("Customer Behaviour")

plot_distributions(df)

st.divider()

# ---------------------------------------------------------
# Distribution Analysis
# ---------------------------------------------------------

st.subheader("Distribution Analysis")

plot_boxplots(df)

st.divider()
# ---------------------------------------------------------
# Correlation Analysis
# ---------------------------------------------------------

st.subheader("Correlation Analysis")

plot_correlation_heatmap(df)

st.divider()

# ---------------------------------------------------------
# Model Insights
# ---------------------------------------------------------

st.subheader("Model Insights")

feature_columns = df.drop(
    columns=["Churn Label"]
).columns

plot_feature_importance(
    model=model,
    feature_names=feature_columns,
    top_n=15
)

st.divider()

# ---------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------

st.subheader("Model Evaluation")

left, right = st.columns(2)

with left:

    plot_roc_curve(
        y_true=y_test,
        y_prob=y_prob
    )

with right:

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred
    )

st.divider()

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

st.subheader("Key Insights")

left, right = st.columns(2)

with left:

    st.container(border=True)

    st.markdown(
        """
### Dataset Highlights

- Customer churn distribution is visualized using class proportions.

- Contract type shows a strong relationship with churn.

- Monthly charges and tenure display noticeable differences between churned and retained customers.

- Feature distributions indicate several variables with clear separation.
"""
    )

with right:

    st.container(border=True)

    st.markdown(
        """
### Model Highlights

- XGBoost provides strong predictive performance.

- SHAP identifies the most influential features driving predictions.

- ROC Curve demonstrates good class separation.

- Confusion Matrix summarizes classification performance across both classes.
"""
    )

st.divider()

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.caption(
    "Analytics Dashboard | ChurnSense AI"
)

