"""
prompts.py

Prompt templates for the Customer Churn Recommendation System.
"""

SYSTEM_PROMPT = """
You are a Senior Customer Retention Strategist at a leading telecom company.

Your responsibility is to analyze the provided customer profile and prepare a customer-specific retention strategy.

Rules:
- Use ONLY the supplied customer information.
- Never invent customer information.
- Never mention AI, SHAP, machine learning, prediction models, feature importance, or algorithms.
- Write in professional business language.
- Return the response in Markdown only.
"""


def build_prompt(
    customer_profile,
    churn_probability,
    risk_level,
    risk_factors,
    protective_factors,
    detail_level="standard"
):
    """
    Build the prompt sent to GPT.
    """

    profile = "\n".join(
        f"- {key}: {value}"
        for key, value in customer_profile.items()
    )

    risk = "\n".join(
        f"- {item['description']}"
        for item in risk_factors
    )

    protection = "\n".join(
        f"- {item['description']}"
        for item in protective_factors
    )

    prompt = f"""
# Customer Profile

{profile}

# Churn Assessment

- Churn Probability: {churn_probability:.2%}
- Risk Level: {risk_level}

# Factors Increasing Churn Risk

{risk}

# Factors Reducing Churn Risk

{protection}

Generate a customer-specific business retention report.

Requirements:

- Base every recommendation ONLY on the information above.
- Do not invent facts.
- Do not mention AI, SHAP, machine learning, feature importance, or prediction models.
- Write in clear business language.
- Return Markdown only.
"""

    # -----------------------------------------------------
    # Short Recommendation (30% - 50%)
    # -----------------------------------------------------

    if detail_level == "short":

        prompt += """

Keep the report under **100 words**.

Use exactly these headings:

## Summary

## Recommended Actions

Provide only **2-3 concise actionable recommendations**.
"""

    # -----------------------------------------------------
    # Standard Recommendation (50% - 70%)
    # -----------------------------------------------------

    elif detail_level == "standard":

        prompt += """

Keep the report under **200 words**.

Use exactly these headings:

## Summary

## Why This Customer May Churn

## Recommended Actions

Provide **3-4 customer-specific recommendations**.

## Expected Business Impact
"""

    # -----------------------------------------------------
    # Detailed Recommendation (70%+)
    # -----------------------------------------------------

    else:

        prompt += """

This customer has a HIGH churn risk.

Provide a detailed business report.

Maximum **300 words**.

Use exactly these headings:

## Summary

## Why This Customer May Churn

## Recommended Actions

Provide **4-5 detailed retention strategies**, ordered from highest to lowest priority.

## Expected Business Impact

Explain how the proposed actions may reduce churn and improve customer retention.
"""

    return prompt