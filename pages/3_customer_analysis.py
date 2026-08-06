"""
pages/3_customer_analysis.py

Customer Analysis
"""

import pandas as pd
import streamlit as st

from llm.pipeline import ChurnRecommendationPipeline

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Customer Analysis",
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

st.title("Customer Analysis")

st.write(
"""
Analyze an individual customer, understand why the model predicts
churn, and inspect the factors influencing the prediction.
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
# Generate Prediction
# ---------------------------------------------------------

with st.spinner("Analyzing customer..."):
    st.write(customer_df.shape)
    st.write(customer_df.head())

    result = pipeline.predict(customer_df)[0]

prediction = result["prediction"]

customer_profile = result["customer_profile"]

risk_factors = result["risk_factors"]

protective_factors = result["protective_factors"]

probability = prediction["probability"]

risk_level = prediction["risk_level"]

# ---------------------------------------------------------
# Prediction Summary
# ---------------------------------------------------------

st.subheader("Prediction Summary")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(

        "Churn Probability",

        f"{probability:.2%}"

    )

with col2:

    st.metric(

        "Risk Level",

        risk_level

    )

with col3:

    label = (
        "Likely to Churn"
        if probability >= 0.5
        else "Likely to Stay"
    )

    st.metric(

        "Prediction",

        label

    )

st.divider()

# ---------------------------------------------------------
# Customer Profile
# ---------------------------------------------------------

st.subheader("Customer Profile")

left, right = st.columns(2)

keys = list(customer_profile.keys())

mid = len(keys) // 2

with left:

    for key in keys[:mid]:

        st.container(border=True)

        st.write(
            f"**{key}**"
        )

        st.write(
            customer_profile[key]
        )

with right:

    for key in keys[mid:]:

        st.container(border=True)

        st.write(
            f"**{key}**"
        )

        st.write(
            customer_profile[key]
        )

st.divider()

# ---------------------------------------------------------
# SHAP Explainability
# ---------------------------------------------------------

st.subheader("SHAP Explainability")

try:

    import matplotlib.pyplot as plt
    import shap

    explainer = pipeline.explainer

    model_input = customer_df.copy()

    if "Churn Label" in model_input.columns:
        model_input = model_input.drop(columns=["Churn Label"])

    shap_values = explainer(model_input)

    fig = plt.figure(figsize=(10, 6))

    shap.plots.waterfall(
        shap_values[0],
        max_display=10,
        show=False
    )

    st.pyplot(fig)

    plt.close(fig)

except Exception as e:

    st.warning(
        f"Unable to generate SHAP waterfall plot.\n\n{e}"
    )

st.divider()

# ---------------------------------------------------------
# Feature Contributions
# ---------------------------------------------------------

st.subheader("Feature Contributions")

left, right = st.columns(2)

# ---------------------------------------------------------
# Risk Factors
# ---------------------------------------------------------

with left:

    st.markdown("#### Top Risk Factors")

    with st.container(border=True):

        if len(risk_factors) == 0:

            st.success(
                "No significant risk factors detected."
            )

        else:

            for factor in risk_factors:

                feature = factor["feature"]
                value = factor["value"]
                impact = factor["impact"]
                direction = factor["direction"]
                st.write(
                    f"**{feature}**"
                )

                st.caption(
                    f"Value: {value}"
                )
                st.caption(
                    f"{direction}: {abs(impact):.3f}"
                )

                st.divider()

# ---------------------------------------------------------
# Protective Factors
# ---------------------------------------------------------

with right:

    st.markdown("#### Top Protective Factors")

    with st.container(border=True):

        if len(protective_factors) == 0:

            st.info(
                "No significant protective factors detected."
            )

        else:

            for factor in protective_factors:

                feature = factor["feature"]
                value = factor["value"]
                impact = factor["impact"]

                st.write(
                    f"**{feature}**"
                )

                st.caption(
                    f"Value: {value}"
                )

                st.caption(
                    f"SHAP Impact: {impact:.3f}"
                )

                st.divider()

st.divider()

# ---------------------------------------------------------
# Business Interpretation
# ---------------------------------------------------------

st.subheader("Business Interpretation")

probability_percent = probability * 100

if probability >= 0.80:

    interpretation = (
        f"This customer has a **very high likelihood of churn "
        f"({probability_percent:.1f}%)**. The model indicates that "
        "multiple risk factors are contributing strongly towards churn. "
        "Immediate retention efforts are recommended."
    )

elif probability >= 0.60:

    interpretation = (
        f"This customer has a **high churn probability "
        f"({probability_percent:.1f}%)**. The identified risk factors "
        "should be addressed through targeted retention strategies."
    )

elif probability >= 0.40:

    interpretation = (
        f"This customer has a **moderate churn probability "
        f"({probability_percent:.1f}%)**. Although several protective "
        "factors exist, proactive engagement may reduce churn risk."
    )

else:

    interpretation = (
        f"This customer has a **low churn probability "
        f"({probability_percent:.1f}%)**. Current customer behaviour "
        "indicates a relatively stable relationship with the business."
    )

st.info(interpretation)

st.divider()

# ---------------------------------------------------------
# Report Information
# ---------------------------------------------------------

st.subheader("Prediction Metadata")

left, right = st.columns(2)

with left:

    st.write("**Model**")
    st.write("XGBoost Classifier")

    st.write("**Explainability**")
    st.write("SHAP TreeExplainer")

with right:

    st.write("**Analysis Generated**")
    st.write(result["generated_at"])

    st.write("**Risk Category**")
    st.write(risk_level)

st.divider()

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.caption(
    "Customer Analysis | ChurnSense AI"
)