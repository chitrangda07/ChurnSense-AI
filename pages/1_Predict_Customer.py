"""
pages/06_predict_customer.py

Predict churn for a new customer using
manual form inputs.
"""

import streamlit as st

from llm.pipeline import ChurnRecommendationPipeline
from utils.form_encoder import encode_customer_form

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Predict Customer",
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
# Load Pipeline
# ---------------------------------------------------------

@st.cache_resource
def load_pipeline():

    return ChurnRecommendationPipeline(
        "models/xgboost_best.pkl"
    )

pipeline = load_pipeline()

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("Predict Customer")

st.write(
"""
Enter customer information below to estimate churn probability
and generate personalized retention recommendations.
"""
)

st.divider()

# ---------------------------------------------------------
# Customer Form
# ---------------------------------------------------------

with st.form("customer_form"):

    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.checkbox(
            "Senior Citizen"
        )

    with col2:

        partner = st.checkbox(
            "Partner"
        )

        dependents = st.checkbox(
            "Dependents"
        )

    st.divider()

    st.subheader("Subscription Details")

    col1, col2 = st.columns(2)

    with col1:

        tenure = st.number_input(
            "Tenure (Months)",
            0,
            100,
            12
        )

        monthly = st.number_input(
            "Monthly Charges",
            0.0,
            500.0,
            70.0
        )

        total = st.number_input(
            "Total Charges",
            0.0,
            10000.0,
            840.0
        )

    with col2:

        cltv = st.number_input(
            "CLTV",
            0.0,
            10000.0,
            3500.0
        )

        paperless = st.checkbox(
            "Paperless Billing"
        )

    st.divider()

    st.subheader("Services")

    col1, col2 = st.columns(2)

    with col1:

        multiple_lines = st.checkbox(
            "Multiple Lines"
        )

        internet = st.selectbox(
            "Internet Service",
            [
                "Fiber Optic",
                "DSL",
                "No"
            ]
        )

        online_security = st.checkbox(
            "Online Security"
        )

        online_backup = st.checkbox(
            "Online Backup"
        )

    with col2:

        device = st.checkbox(
            "Device Protection"
        )

        tech = st.checkbox(
            "Tech Support"
        )

        tv = st.checkbox(
            "Streaming TV"
        )

        movies = st.checkbox(
            "Streaming Movies"
        )

    st.divider()

    st.subheader("Contract & Payment")

    col1, col2 = st.columns(2)

    with col1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-Month",
                "One Year",
                "Two Year"
            ]
        )

    with col2:

        payment = st.selectbox(
            "Payment Method",
            [
                "Credit Card (automatic)",
                "Electronic Check",
                "Mailed Check",
                "Bank Transfer (automatic)"
            ]
        )

    predict = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )
# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if predict:

    form_data = {

        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,

        "Tenure Months": tenure,
        "Monthly Charges": monthly,
        "Total Charges": total,
        "CLTV": cltv,
        "Paperless Billing": paperless,

        "Multiple Lines": multiple_lines,
        "Internet Service": internet,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device,
        "Tech Support": tech,
        "Streaming TV": tv,
        "Streaming Movies": movies,

        "Contract": contract,
        "Payment Method": payment

    }

    customer_df = encode_customer_form(form_data)

    with st.spinner("Predicting customer churn..."):

        result = pipeline.predict_customer(customer_df)

    prediction = result["prediction"]

    probability = prediction["probability"]

    risk_level = prediction["risk_level"]

    risk_factors = result["risk_factors"]

    protective_factors = result["protective_factors"]

    business_report = result["business_report"]

    st.divider()

    # -----------------------------------------------------
    # Prediction Summary
    # -----------------------------------------------------

    st.subheader("Prediction Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    with col2:

        st.metric(
            "Confidence",
            prediction["confidence"]
        )

    with col3:

        st.metric(
            "Risk Level",
            risk_level
        )

    st.divider()

    # -----------------------------------------------------
    # Key Drivers
    # -----------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("Key Risk Factors")

        with st.container(border=True):

            if len(risk_factors) == 0:

                st.success(
                    "No major churn drivers detected."
                )

            else:

                for factor in risk_factors:

                    st.markdown(
                        f"**{factor['feature']}**"
                    )

                    st.write(
                        factor["description"]
                    )

                    st.divider()

    with right:

        st.subheader("Key Protective Factors")

        with st.container(border=True):

            if len(protective_factors) == 0:

                st.info(
                    "No major protective factors detected."
                )

            else:

                for factor in protective_factors:

                    st.markdown(
                        f"**{factor['feature']}**"
                    )

                    st.write(
                        factor["description"]
                    )

                    st.divider()

    st.divider()

    # -----------------------------------------------------
    # AI Recommendation
    # -----------------------------------------------------

    st.subheader("AI Retention Strategy")

    with st.container(border=True):

        st.markdown(
            business_report
        )

    st.divider()

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Prediction Model**")
        st.write("XGBoost")

        st.write("**Explainability**")
        st.write("SHAP Feature Attribution")

    with col2:

        st.write("**LLM**")
        st.write("GPT-5 mini")

        st.write("**Generated At**")
        st.write(result["generated_at"])

st.divider()

st.caption(
    "Predict Customer | ChurnSense AI"
)