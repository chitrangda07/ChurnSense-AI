"""
pages/4_Ai_recommender.py

AI Business Recommendations
"""

import pandas as pd
import streamlit as st

from llm.pipeline import ChurnRecommendationPipeline

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Recommendations",
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

st.title("AI Recommendations")

st.write(
"""
Generate personalized customer retention strategies using GPT-5 mini.
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
# Load Pipeline
# ---------------------------------------------------------

@st.cache_resource
def load_pipeline():

    return ChurnRecommendationPipeline(
        "models/xgboost_best.pkl"
    )

pipeline = load_pipeline()

# ---------------------------------------------------------
# Customer Selection
# ---------------------------------------------------------

st.subheader("Customer Selection")

customer_index = st.selectbox(

    "Select Customer",

    options=df.index,

    format_func=lambda x: f"Customer {x}"

)

customer_df = df.iloc[[customer_index]]

# ---------------------------------------------------------
# Generate Recommendation
# ---------------------------------------------------------

if st.button(
    "Generate AI Recommendation",
    use_container_width=True
):

    with st.spinner(
        "Generating business recommendations..."
    ):

        result = pipeline.predict(customer_df)[0]

    prediction = result["prediction"]

    business_report = result["business_report"]

    st.divider()

    # -----------------------------------------------------
    # Prediction Summary
    # -----------------------------------------------------

    st.subheader("Prediction Summary")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(

            "Churn Probability",

            f"{prediction['probability']:.2%}"

        )

    with col2:

        st.metric(

            "Risk Level",

            prediction["risk_level"]

        )

    st.divider()

    # -----------------------------------------------------
    # AI Report
    # -----------------------------------------------------

    st.subheader("Business Recommendation")

    st.markdown(
        business_report
    )

    st.divider()

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.write("**LLM**")
        st.write("GPT-5 mini")

        st.write("**Generated At**")
        st.write(result["generated_at"])

    with right:

        st.write("**Model**")
        st.write("XGBoost")

        st.write("**Explainability**")
        st.write("SHAP")

st.divider()

st.caption(
    "AI Recommendations | ChurnSense AI"
)