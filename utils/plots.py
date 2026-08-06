import streamlit as st


def plot_kpi_cards(
    df,
    accuracy,
    roc_auc,
    f1_score
):
    """
    Displays the top KPI cards for the analytics dashboard.
    """

    churn_rate = df["Churn Label"].mean() * 100
    total_customers = len(df)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Customers",
            value=f"{total_customers:,}"
        )

    with col2:
        st.metric(
            label="Churn Rate",
            value=f"{churn_rate:.2f}%"
        )

    with col3:
        st.metric(
            label="Accuracy",
            value=f"{accuracy:.2%}"
        )

    with col4:
        st.metric(
            label="ROC-AUC",
            value=f"{roc_auc:.3f}"
        )

    with col5:
        st.metric(
            label="F1 Score",
            value=f"{f1_score:.3f}"
        )

    st.markdown("---")

import plotly.express as px
def plot_churn_distribution(df):
    """
    Displays the churn distribution as a donut chart.
    """

    churn_counts = (
        df["Churn Label"]
        .replace({0: "Retained", 1: "Churned"})
        .value_counts()
        .reset_index()
    )

    churn_counts.columns = ["Status", "Customers"]

    fig = px.pie(
        churn_counts,
        names="Status",
        values="Customers",
        hole=0.55,
        title="Customer Churn Distribution",
        color="Status",
        color_discrete_map={
            "Retained": "#10B981",
            "Churned": "#EF4444"
        }
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Customers: %{value}<extra></extra>"
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        title_x=0.5,
        margin=dict(t=60, b=20, l=20, r=20),
        showlegend=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
import pandas as pd

def plot_contract_analysis(df):

    """
    Displays Contract Type vs Churn grouped bar chart.
    """

    contract = pd.Series("Month-to-Month", index=df.index)

    contract[df["Contract_One year"] == 1] = "One Year"
    contract[df["Contract_Two year"] == 1] = "Two Year"

    churn = df["Churn Label"].replace(
        {
            0: "Retained",
            1: "Churned"
        }
    )

    contract_df = pd.DataFrame(
        {
            "Contract": contract,
            "Churn": churn
        }
    )

    grouped = (
        contract_df
        .groupby(["Contract", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        grouped,
        x="Contract",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Contract Type vs Customer Churn",
        color_discrete_map={
            "Retained": "#10B981",
            "Churned": "#EF4444"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        title_x=0.5,
        height=450,
        xaxis_title="Contract Type",
        yaxis_title="Customers",
        legend_title="Status",
        margin=dict(t=60, l=20, r=20, b=20)
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>%{legendgroup}<br>Customers=%{y}<extra></extra>"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def plot_distributions(df):

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Monthly Charges",
            nbins=35,
            title="Monthly Charges Distribution",
            color_discrete_sequence=["#3B82F6"]
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            title_x=0.5
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            df,
            x="Tenure Months",
            nbins=30,
            title="Customer Tenure Distribution",
            color_discrete_sequence=["#10B981"]
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            title_x=0.5
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

def plot_boxplots(df):
    """
    Displays boxplots comparing feature distributions
    for churned vs retained customers.
    """

    plot_df = df.copy()

    plot_df["Churn"] = plot_df["Churn Label"].replace(
        {
            0: "Retained",
            1: "Churned"
        }
    )

    col1, col2, col3 = st.columns(3)

    # -------------------------------
    # Monthly Charges
    # -------------------------------

    with col1:

        fig = px.box(
            plot_df,
            x="Churn",
            y="Monthly Charges",
            color="Churn",
            title="Monthly Charges vs Churn",
            color_discrete_map={
                "Retained": "#10B981",
                "Churned": "#EF4444"
            },
            points="outliers"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            showlegend=False,
            title_x=0.5,
            margin=dict(t=50, l=20, r=20, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------------
    # Tenure
    # -------------------------------

    with col2:

        fig = px.box(
            plot_df,
            x="Churn",
            y="Tenure Months",
            color="Churn",
            title="Tenure vs Churn",
            color_discrete_map={
                "Retained": "#10B981",
                "Churned": "#EF4444"
            },
            points="outliers"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            showlegend=False,
            title_x=0.5,
            margin=dict(t=50, l=20, r=20, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------------
    # Total Charges
    # -------------------------------

    with col3:

        fig = px.box(
            plot_df,
            x="Churn",
            y="Total Charges",
            color="Churn",
            title="Total Charges vs Churn",
            color_discrete_map={
                "Retained": "#10B981",
                "Churned": "#EF4444"
            },
            points="outliers"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            showlegend=False,
            title_x=0.5,
            margin=dict(t=50, l=20, r=20, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

def plot_business_insights(df):
    """
    Displays Internet Service and Payment Method analysis.
    """

    plot_df = df.copy()

    # -----------------------------
    # Internet Service
    # -----------------------------

    internet = pd.Series("DSL", index=plot_df.index)

    internet[plot_df["Internet Service_Fiber optic"] == 1] = "Fiber Optic"
    internet[plot_df["Internet Service_No"] == 1] = "No Internet"

    # -----------------------------
    # Payment Method
    # -----------------------------

    payment = pd.Series("Bank Transfer", index=plot_df.index)

    payment[
        plot_df["Payment Method_Credit card (automatic)"] == 1
    ] = "Credit Card"

    payment[
        plot_df["Payment Method_Electronic check"] == 1
    ] = "Electronic Check"

    payment[
        plot_df["Payment Method_Mailed check"] == 1
    ] = "Mailed Check"

    churn = plot_df["Churn Label"].replace(
        {
            0: "Retained",
            1: "Churned"
        }
    )

    internet_df = pd.DataFrame(
        {
            "Internet Service": internet,
            "Churn": churn
        }
    )

    payment_df = pd.DataFrame(
        {
            "Payment Method": payment,
            "Churn": churn
        }
    )

    col1, col2 = st.columns(2)

    # ---------------------------------
    # Internet Service
    # ---------------------------------

    with col1:

        grouped = (
            internet_df
            .groupby(
                ["Internet Service", "Churn"]
            )
            .size()
            .reset_index(name="Customers")
        )

        fig = px.bar(
            grouped,
            x="Internet Service",
            y="Customers",
            color="Churn",
            barmode="group",
            title="Internet Service vs Churn",
            color_discrete_map={
                "Retained": "#10B981",
                "Churned": "#EF4444"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
            title_x=0.5
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------------------------
    # Payment Method
    # ---------------------------------

    with col2:

        grouped = (
            payment_df
            .groupby(
                ["Payment Method", "Churn"]
            )
            .size()
            .reset_index(name="Customers")
        )

        fig = px.bar(
            grouped,
            x="Payment Method",
            y="Customers",
            color="Churn",
            barmode="group",
            title="Payment Method vs Churn",
            color_discrete_map={
                "Retained": "#10B981",
                "Churned": "#EF4444"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
            title_x=0.5
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df):
    """
    Displays correlation heatmap.
    """

    numeric_df = df.select_dtypes(include="number")

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap",
        fontsize=16,
        pad=20
    )

    st.pyplot(fig)

def plot_feature_importance(model, feature_names, top_n=15):
    """
    Displays XGBoost feature importance.
    """

    feature_names = list(feature_names)
    importances = list(model.feature_importances_)

    n = min(len(feature_names), len(importances))

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names[:n],
            "Importance": importances[:n]
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(top_n)
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Important Features",
        color="Importance",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        template="plotly_dark",
        height=600,
        title_x=0.5,
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        margin=dict(t=60, l=20, r=20, b=20)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

def plot_confusion_matrix(
    y_true,
    y_pred
):
    """
    Displays confusion matrix.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Retained", "Churned"],
        y=["Retained", "Churned"],
        colorscale="Blues",
        showscale=True
    )

    fig.update_layout(
        title="Confusion Matrix",
        template="plotly_dark",
        title_x=0.5,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

from sklearn.metrics import (
    roc_curve,
    auc
)

import plotly.graph_objects as go


def plot_roc_curve(
    y_true,
    y_prob
):
    """
    Displays ROC Curve.
    """

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"AUC = {roc_auc:.3f}",
            line=dict(
                color="#3B82F6",
                width=3
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(
                dash="dash",
                color="gray"
            ),
            showlegend=False
        )
    )

    fig.update_layout(
        title="ROC Curve",
        template="plotly_dark",
        title_x=0.5,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


