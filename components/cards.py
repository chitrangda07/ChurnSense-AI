"""
cards.py

Reusable UI components for the ChurnSense AI dashboard.
"""

import streamlit as st


# -------------------------------------------------------
# Metric Card
# -------------------------------------------------------

def metric_card(title: str,
                value: str,
                subtitle: str = "",
                color: str = "#3B82F6"):
    """
    Display a custom metric card.
    """

    st.markdown(
        f"""
        <div class="custom-card">

            <div class="metric-title">
                {title}
            </div>

            <div class="metric-value"
                 style="color:{color};">

                {value}

            </div>

            <div class="metric-sub">

                {subtitle}

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------
# Information Card
# -------------------------------------------------------

def info_card(title: str, body: str):
    """
    Display a simple information card.
    """

    st.markdown(
        f"""
        <div class="custom-card">

            <h3>{title}</h3>

            <p style="
                color:#CBD5E1;
                line-height:1.8;
            ">

            {body}

            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------
# Bullet Card
# -------------------------------------------------------

def bullet_card(title: str,
                items: list):
    """
    Display a card with bullet points.
    """

    bullets = ""

    for item in items:

        bullets += f"<li>{item}</li>"

    st.markdown(
        f"""
        <div class="custom-card">

            <h3>{title}</h3>

            <ul style="
                color:#CBD5E1;
                line-height:2;
            ">

            {bullets}

            </ul>

        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------
# Divider
# -------------------------------------------------------

def section_divider(title: str):
    """
    Display a section heading with a divider.
    """

    st.markdown(
        f"""
        <h2 style="
            margin-top:10px;
            margin-bottom:10px;
        ">
            {title}
        </h2>

        <hr style="
            border:1px solid #334155;
            margin-bottom:25px;
        ">
        """,
        unsafe_allow_html=True
    )