"""
pages/5_Batch_Prediction.py

Batch Prediction
"""

import pandas as pd
import streamlit as st

from llm.pipeline import ChurnRecommendationPipeline

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Batch Prediction",
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

st.title("Batch Prediction")

st.write(
"""
Upload a processed customer dataset to generate churn predictions
and AI-powered retention recommendations for multiple customers.
"""
)

st.divider()

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
# File Upload
# ---------------------------------------------------------

st.subheader("Upload Dataset")

uploaded_file = st.file_uploader(

    "Upload Processed CSV",

    type=["csv"]

)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success(
        f"{len(df)} customer records loaded successfully."
    )

    st.divider()

    # -----------------------------------------------------
    # Preview
    # -----------------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    if st.button(

        "Run Batch Prediction",

        use_container_width=True

    ):

        with st.spinner(

            "Generating predictions..."

        ):

            reports = pipeline.predict(df)

        results = []

        for report in reports:

            prediction = report["prediction"]

            results.append({

                "Probability":
                    prediction["probability"],

                "Risk Level":
                    prediction["risk_level"],

                "Generated At":
                    report["generated_at"]

            })

        result_df = pd.DataFrame(results)

        st.success(

            "Batch prediction completed."

        )

        st.divider()

        # -------------------------------------------------
        # Results
        # -------------------------------------------------

        st.subheader("Prediction Results")

        st.dataframe(

            result_df,

            use_container_width=True

        )

        st.divider()

        # -------------------------------------------------
        # Download
        # -------------------------------------------------

        csv = result_df.to_csv(

            index=False

        ).encode("utf-8")

        st.download_button(

            label="Download Predictions",

            data=csv,

            file_name="batch_predictions.csv",

            mime="text/csv",

            use_container_width=True

        )

else:

    st.info(

        "Upload a processed CSV file to begin batch prediction."

    )

st.divider()

st.caption(
    "Batch Prediction | ChurnSense AI"
)