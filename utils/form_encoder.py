"""
form_encoder.py

Converts user-friendly form inputs into the
feature vector expected by the trained XGBoost model.
"""

import pandas as pd


def encode_customer_form(form_data: dict) -> pd.DataFrame:
    """
    Convert form inputs into the processed feature vector.

    Parameters
    ----------
    form_data : dict
        Dictionary containing user inputs from the Streamlit form.

    Returns
    -------
    pd.DataFrame
        Single-row dataframe ready for prediction.
    """

    features = {

        # -------------------------------------------------
        # Basic Information
        # -------------------------------------------------

        "Gender": 1 if form_data["Gender"] == "Male" else 0,

        "Senior Citizen": 1 if form_data["Senior Citizen"] else 0,

        "Partner": 1 if form_data["Partner"] else 0,

        "Dependents": 1 if form_data["Dependents"] else 0,

        # -------------------------------------------------
        # Numerical Features
        # -------------------------------------------------

        "Tenure Months": form_data["Tenure Months"],

        "Paperless Billing": (
            1 if form_data["Paperless Billing"] else 0
        ),

        "Monthly Charges": form_data["Monthly Charges"],

        "Total Charges": form_data["Total Charges"],

        "CLTV": form_data["CLTV"],

        # -------------------------------------------------
        # Multiple Lines
        # -------------------------------------------------

        "Multiple Lines_Yes": (
            1 if form_data["Multiple Lines"] else 0
        ),

        # -------------------------------------------------
        # Internet Service
        # -------------------------------------------------

        "Internet Service_Fiber optic": (
            1 if form_data["Internet Service"] == "Fiber Optic"
            else 0
        ),

        "Internet Service_No": (
            1 if form_data["Internet Service"] == "No"
            else 0
        ),

        # -------------------------------------------------
        # Online Services
        # -------------------------------------------------

        "Online Security_Yes": (
            1 if form_data["Online Security"] else 0
        ),

        "Online Backup_Yes": (
            1 if form_data["Online Backup"] else 0
        ),

        "Device Protection_Yes": (
            1 if form_data["Device Protection"] else 0
        ),

        "Tech Support_Yes": (
            1 if form_data["Tech Support"] else 0
        ),

        "Streaming TV_Yes": (
            1 if form_data["Streaming TV"] else 0
        ),

        "Streaming Movies_Yes": (
            1 if form_data["Streaming Movies"] else 0
        ),

        # -------------------------------------------------
        # Contract
        # -------------------------------------------------

        "Contract_One year": (
            1 if form_data["Contract"] == "One Year"
            else 0
        ),

        "Contract_Two year": (
            1 if form_data["Contract"] == "Two Year"
            else 0
        ),

        # -------------------------------------------------
        # Payment Method
        # -------------------------------------------------

        "Payment Method_Credit card (automatic)": (
            1
            if form_data["Payment Method"]
            == "Credit Card (automatic)"
            else 0
        ),

        "Payment Method_Electronic check": (
            1
            if form_data["Payment Method"]
            == "Electronic Check"
            else 0
        ),

        "Payment Method_Mailed check": (
            1
            if form_data["Payment Method"]
            == "Mailed Check"
            else 0
        )
    }

    # -------------------------------------------------
    # Engineered Feature
    # -------------------------------------------------

    features["HighCharge_MonthlyContract"] = int(

        form_data["Monthly Charges"] > 70

        and

        form_data["Contract"] == "Month-to-Month"

    )

    return pd.DataFrame([features])