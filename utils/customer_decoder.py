"""
customer_decoder.py

Convert processed (encoded) customer data into
business-friendly values for the dashboard and reports.
"""


def decode_customer(customer):
    """
    Convert one processed customer row into readable values.

    Parameters
    ----------
    customer : pd.Series

    Returns
    -------
    dict
    """

    profile = {}

    # ----------------------------
    # Gender
    # ----------------------------

    profile["Gender"] = (
        "Male"
        if customer["Gender"] == 1
        else "Female"
    )

    # ----------------------------
    # Senior Citizen
    # ----------------------------

    profile["Senior Citizen"] = (
        "Yes"
        if customer["Senior Citizen"] == 1
        else "No"
    )

    # ----------------------------
    # Partner
    # ----------------------------

    profile["Partner"] = (
        "Yes"
        if customer["Partner"] == 1
        else "No"
    )

    # ----------------------------
    # Dependents
    # ----------------------------

    profile["Dependents"] = (
        "Yes"
        if customer["Dependents"] == 1
        else "No"
    )

    # ----------------------------
    # Phone Service
    # ----------------------------

    profile["Phone Service"] = (
        "Yes"
        if customer["Phone Service"] == 1
        else "No"
    )

    # ----------------------------
    # Internet Service
    # ----------------------------

    if customer["Internet Service_No"] == 1:
        profile["Internet Service"] = "No Internet"

    elif customer["Internet Service_Fiber optic"] == 1:
        profile["Internet Service"] = "Fiber Optic"

    else:
        profile["Internet Service"] = "DSL"

    # ----------------------------
    # Contract
    # ----------------------------

    if customer["Contract_Two year"] == 1:

        profile["Contract"] = "Two Year"

    elif customer["Contract_One year"] == 1:

        profile["Contract"] = "One Year"

    else:

        profile["Contract"] = "Month-to-Month"

    # ----------------------------
    # Payment Method
    # ----------------------------

    if customer["Payment Method_Electronic check"] == 1:

        profile["Payment Method"] = "Electronic Check"

    elif customer["Payment Method_Credit card (automatic)"] == 1:

        profile["Payment Method"] = "Credit Card"

    elif customer["Payment Method_Mailed check"] == 1:

        profile["Payment Method"] = "Mailed Check"

    else:

        profile["Payment Method"] = "Bank Transfer"

    # ----------------------------
    # Numerical
    # ----------------------------

    profile["Tenure"] = f"{int(customer['Tenure Months'])} Months"

    profile["Monthly Charges"] = f"${customer['Monthly Charges']:.2f}"

    profile["Total Charges"] = f"${customer['Total Charges']:.2f}"

    profile["Customer Lifetime Value"] = (
        f"${customer['CLTV']:.0f}"
    )

    return profile