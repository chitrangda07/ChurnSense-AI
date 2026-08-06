"""
pages/1_home.py

Landing page for ChurnSense AI
"""

import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="ChurnSense AI",
    layout="wide"
)

# ---------------------------------------------------------
# Load Custom CSS
# ---------------------------------------------------------

with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# Hero Section
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero-container">

    <h1 class="hero-title">
        ChurnSense AI
    </h1>

    <p class="hero-subtitle">
        Explainable Customer Churn Intelligence Platform
    </p>

    <p class="hero-description">
        ChurnSense AI combines Machine Learning, Explainable AI (SHAP),
        and GPT-5 mini to predict customer churn, explain the reasons
        behind every prediction, and generate personalized business
        retention strategies.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------------
# Project Highlights
# ---------------------------------------------------------

st.subheader("Project Highlights")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", "XGBoost")

with col2:
    st.metric("ROC-AUC", "0.851")

with col3:
    st.metric("Accuracy", "77.7%")

with col4:
    st.metric("Explainability", "SHAP")

st.divider()

# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------

st.subheader("Project Overview")

st.write(
"""
Customer churn is one of the biggest challenges faced by subscription-based businesses.
While predicting churn is valuable, understanding *why* a customer is likely to leave
is equally important.

ChurnSense AI provides an end-to-end decision support platform that:

- Predicts customer churn using an optimized XGBoost model.
- Explains every prediction using SHAP values.
- Generates business-focused retention strategies using GPT-5 mini.
- Supports both single-customer and batch predictions.
"""
)

st.divider()

# ---------------------------------------------------------
# Platform Modules
# ---------------------------------------------------------

st.subheader("Platform Modules")

c1, c2 = st.columns(2)

with c1:

    with st.container(border=True):

        st.markdown("### Analytics Dashboard")

        st.write(
        """
        Explore customer behaviour through
        interactive visualizations including:

        • Churn Distribution

        • Contract Analysis

        • Feature Importance

        • Model Performance

        • Correlation Analysis
        """
        )

    with st.container(border=True):

        st.markdown("### Customer Analysis")

        st.write(
        """
        Analyze an individual customer with:

        • Churn Probability

        • SHAP Explainability

        • Risk Factors

        • Protective Factors
        """
        )

with c2:

    with st.container(border=True):

        st.markdown("### AI Recommendations")

        st.write(
        """
        Generate business-focused retention
        recommendations using GPT-5 mini based
        on customer risk profile.
        """
        )

    with st.container(border=True):

        st.markdown("### Batch Prediction")

        st.write(
        """
        Upload a processed dataset,
        predict churn for thousands of
        customers, and export the results.
        """
        )

st.divider()

# ---------------------------------------------------------
# Technology Stack
# ---------------------------------------------------------

st.subheader("Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:

    with st.container(border=True):

        st.markdown("### Machine Learning")

        st.write(
        """
        - XGBoost

        - Scikit-learn

        - Feature Engineering

        - Hyperparameter Tuning
        """
        )

with col2:

    with st.container(border=True):

        st.markdown("### Explainable AI")

        st.write(
        """
        - SHAP

        - TreeExplainer

        - Global Explainability

        - Local Explainability
        """
        )

with col3:

    with st.container(border=True):

        st.markdown("### LLM Layer")

        st.write(
        """
        - GPT-5 mini

        - OpenAI API

        - Prompt Engineering

        - Business Recommendations
        """
        )

st.divider()

# ---------------------------------------------------------
# Workflow
# ---------------------------------------------------------

st.subheader("System Workflow")

st.image(
    "assets/workflow.png",
    use_container_width=True
)

st.divider()

# ---------------------------------------------------------
# Dataset Summary
# ---------------------------------------------------------

st.subheader("Dataset")

left, right = st.columns([2, 1])

with left:

    st.write(
    """
    **Dataset Used**

    - Telco Customer Churn Dataset

    - 7,043 Customers

    - 25 Engineered Features

    - Binary Classification Problem

    - Target Variable: Churn Label
    """
    )

with right:

    st.info(
        """
        **Model Pipeline**

        Customer Data

        ↓

        Feature Engineering

        ↓

        XGBoost

        ↓

        SHAP

        ↓

        GPT-5 mini
        """
    )

st.divider()

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.caption(
    "ChurnSense AI • Explainable Machine Learning • XGBoost • SHAP • GPT-5 mini"
)