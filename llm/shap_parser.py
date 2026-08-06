"""
Converts SHAP explanations into business-friendly insights
that can be consumed by an LLM.
"""

import pandas as pd


def determine_risk_level(probability: float) -> str:
    """
    Convert churn probability into a business risk level.
    """

    if probability >= 0.80:
        return "Critical"
    elif probability >= 0.60:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    else:
        return "Low"

def translate_feature(feature, value):
    """
    Convert encoded feature values into business-friendly language.
    """

    if feature == "Tenure Months":
        return f"Customer has been with the company for {int(value)} months."

    elif feature == "Monthly Charges":
        return f"Monthly charges are ${value:.2f}."

    elif feature == "Total Charges":
        return f"Total amount paid by the customer is ${value:.2f}."

    elif feature == "CLTV":
        return f"Customer Lifetime Value (CLTV) is {value:.0f}."

    elif feature == "Partner":
        return (
            "Customer has a partner."
            if value == 1
            else "Customer does not have a partner."
        )

    elif feature == "Dependents":
        return (
            "Customer has dependents."
            if value == 1
            else "Customer has no dependents."
        )

    elif feature == "Paperless Billing":
        return (
            "Customer uses paperless billing."
            if value == 1
            else "Customer does not use paperless billing."
        )

    elif feature == "Senior Citizen":
        return (
            "Customer is a senior citizen."
            if value == 1
            else "Customer is not a senior citizen."
        )

    elif feature == "Gender":
        return (
            "Customer is Male."
            if value == 1
            else "Customer is Female."
        )

    elif feature == "Contract_Two year":
        return (
            "Customer is on a two-year contract."
            if value == 1
            else "Customer is not on a two-year contract."
        )

    elif feature == "Contract_One year":
        return (
            "Customer is on a one-year contract."
            if value == 1
            else "Customer is not on a one-year contract."
        )

    elif feature == "Internet Service_Fiber optic":
        return (
            "Customer uses Fiber Optic internet."
            if value == 1
            else "Customer does not use Fiber Optic internet."
        )

    elif feature == "Internet Service_No":
        return (
            "Customer does not subscribe to internet service."
            if value == 1
            else "Customer subscribes to internet service."
        )

    elif feature == "Payment Method_Electronic check":
        return (
            "Customer pays using Electronic Check."
            if value == 1
            else "Customer does not use Electronic Check."
        )

    elif feature == "Payment Method_Credit card (automatic)":
        return (
            "Customer pays using automatic Credit Card."
            if value == 1
            else "Customer does not use automatic Credit Card."
        )

    elif feature == "Payment Method_Mailed check":
        return (
            "Customer pays using Mailed Check."
            if value == 1
            else "Customer does not use Mailed Check."
        )

    elif feature == "HighCharge_MonthlyContract":
        return (
            "Customer has high monthly charges while on a month-to-month contract."
            if value == 1
            else "Customer is not in the high-charge month-to-month segment."
        )

    elif feature == "Tech Support_Yes":
        return (
            "Customer subscribes to Tech Support."
            if value == 1
            else "Customer does not subscribe to Tech Support."
        )

    elif feature == "Online Security_Yes":
        return (
            "Customer has Online Security."
            if value == 1
            else "Customer does not have Online Security."
        )

    elif feature == "Device Protection_Yes":
        return (
            "Customer has Device Protection."
            if value == 1
            else "Customer does not have Device Protection."
        )

    return f"{feature}: {value}"

def parse_shap_values(
    shap_values,
    customer_row,
    feature_names,
    top_n=5
):
    """
    Returns structured risk and protective factors.
    """

    feature_data = []

    for feature, shap_value in zip(feature_names, shap_values):

        feature_data.append({

        "feature": feature,

        "description": translate_feature(
            feature,
            customer_row[feature]
        ),

        "value": customer_row[feature],

        "shap_value": float(shap_value),

        # Numeric value used for plotting/display
        "impact": float(shap_value),

        # Human-readable direction
        "direction": (
            "Increase"
            if shap_value > 0
            else "Decrease"
        )

    })

    risk_factors = sorted(
        [f for f in feature_data if f["shap_value"] > 0],
        key=lambda x: x["shap_value"],
        reverse=True
    )[:top_n]

    protective_factors = sorted(
        [f for f in feature_data if f["shap_value"] < 0],
        key=lambda x: x["shap_value"]
    )[:top_n]

    return risk_factors, protective_factors



def build_customer_profile(customer_row):
    """
    Returns a concise customer profile for the LLM.
    """

    return {

        "Tenure (Months)": int(customer_row["Tenure Months"]),

        "Monthly Charges": round(customer_row["Monthly Charges"], 2),

        "Total Charges": round(customer_row["Total Charges"], 2),

        "Customer Lifetime Value": round(customer_row["CLTV"], 2),

        "Partner": (
            "Yes"
            if customer_row["Partner"]
            else "No"
        ),

        "Dependents": (
            "Yes"
            if customer_row["Dependents"]
            else "No"
        ),

        "Paperless Billing": (
            "Yes"
            if customer_row["Paperless Billing"]
            else "No"
        ),

        "Fiber Internet": (
            "Yes"
            if customer_row["Internet Service_Fiber optic"]
            else "No"
        ),

        "Two-Year Contract": (
            "Yes"
            if customer_row["Contract_Two year"]
            else "No"
        ),

        "Electronic Check": (
            "Yes"
            if customer_row["Payment Method_Electronic check"]
            else "No"
        )
    }