"""
pipeline.py

End-to-end pipeline for generating customer churn predictions
and AI-powered business recommendations.
"""

import logging
from datetime import datetime

import joblib
import pandas as pd
import shap

from llm.prompts import build_prompt
from llm.recommender import ChurnRecommender
from llm.shap_parser import (
    determine_risk_level,
    parse_shap_values,
    build_customer_profile,
)

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------

class ChurnRecommendationPipeline:
    """
    End-to-end pipeline for customer churn prediction,
    SHAP explainability, and GPT-powered recommendations.
    """

    def __init__(self, model_path: str):
        """
        Initialize the recommendation pipeline.

        Parameters
        ----------
        model_path : str
            Path to the trained XGBoost model.
        """

        self.model = joblib.load(model_path)

        self.explainer = shap.TreeExplainer(self.model)

        self.recommender = ChurnRecommender()

        logger.info("Pipeline initialized successfully.")

    # -----------------------------------------------------

    def _get_shap_values(
        self,
        customer_df: pd.DataFrame
    ):
        """
        Returns SHAP values while supporting
        multiple SHAP versions.
        """

        try:

            shap_output = self.explainer(customer_df)

            if hasattr(shap_output, "values"):
                return shap_output.values

            return shap_output

        except Exception:

            return self.explainer.shap_values(customer_df)

    # -----------------------------------------------------

    def predict(
        self,
        customer_df: pd.DataFrame
    ):
        """
        Predict churn for one or more customers.

        Parameters
        ----------
        customer_df : pd.DataFrame
            Processed customer dataframe.

        Returns
        -------
        list
            List containing prediction reports.
        """

        results = []

        # ---------------------------------------------
        # Remove target column if present
        # ---------------------------------------------

        customer_df = customer_df.copy()

        if "Churn Label" in customer_df.columns:

            customer_df = customer_df.drop(
                columns=["Churn Label"]
            )

        # ---------------------------------------------
        # Model Prediction
        # ---------------------------------------------

        probabilities = self.model.predict_proba(
            customer_df
        )[:, 1]

        shap_values = self._get_shap_values(
            customer_df
        )

        # ---------------------------------------------
        # Process Each Customer
        # ---------------------------------------------

        for idx in range(len(customer_df)):

            customer_row = customer_df.iloc[idx]

            probability = float(probabilities[idx])

            confidence = max(
                probability,
                1 - probability
            )

            risk_level = determine_risk_level(
                probability
            )

            sample_shap = shap_values[idx]

            risk_factors, protective_factors = parse_shap_values(
                sample_shap,
                customer_row,
                customer_df.columns.tolist()
            )

            customer_profile = build_customer_profile(
                customer_row
            )

            # ---------------------------------------------
# Generate Business Recommendation
# ---------------------------------------------

            if probability < 0.30:

                business_report = """
            ## Customer Status

            This customer has a **low likelihood of churn**.

            No immediate retention intervention is recommended.

            Continue regular customer engagement and monitor customer satisfaction through standard business processes.
            """

            elif probability < 0.50:

                prompt = build_prompt(
                    customer_profile=customer_profile,
                    churn_probability=probability,
                    risk_level=risk_level,
                    risk_factors=risk_factors,
                    protective_factors=protective_factors,
                    detail_level="short"
                )

                business_report = self.recommender.generate_report(prompt)

            elif probability < 0.70:

                prompt = build_prompt(
                    customer_profile=customer_profile,
                    churn_probability=probability,
                    risk_level=risk_level,
                    risk_factors=risk_factors,
                    protective_factors=protective_factors,
                    detail_level="standard"
                )

                business_report = self.recommender.generate_report(prompt)

            else:

                prompt = build_prompt(
                    customer_profile=customer_profile,
                    churn_probability=probability,
                    risk_level=risk_level,
                    risk_factors=risk_factors,
                    protective_factors=protective_factors,
                    detail_level="detailed"
                )

                business_report = self.recommender.generate_report(prompt)

            results.append(

                {

                    "generated_at": datetime.now().isoformat(),

                    "model_info": {

                        "model": "XGBoost",

                        "version": "1.0",

                        "explainer": "SHAP",

                        "llm": "gpt-5-mini"

                    },

                    "prediction": {

                        "probability": round(
                            probability,
                            4
                        ),

                        "confidence": f"{confidence * 100:.2f}%",

                        "risk_level": risk_level

                    },

                    "customer_profile": customer_profile,

                    "risk_factors": risk_factors,

                    "protective_factors": protective_factors,

                    "business_report": business_report

                }

            )

        logger.info(
            f"Generated {len(results)} recommendation(s)."
        )

        return results

    # -----------------------------------------------------

    def predict_customer(
        self,
        customer_df: pd.DataFrame
    ):
        """
        Convenience method for predicting
        a single customer.

        Returns
        -------
        dict
            Prediction report.
        """

        return self.predict(customer_df)[0]