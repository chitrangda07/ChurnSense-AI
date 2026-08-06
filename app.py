import streamlit as st

st.set_page_config(
    page_title="ChurnSense AI",
    layout="wide"
)

pg = st.navigation([
    st.Page("pages/1_home.py", title="Home"),
    st.Page("pages/2_Analysis_dashboard.py", title="Analytics Dashboard"),
    st.Page("pages/3_customer_analysis.py", title="Customer Analysis"),
    st.Page("pages/1_Predict_Customer.py", title="Predict Customer"),
        st.Page("pages/4_Ai_recommender.py", title="AI Recommender"),
    st.Page("pages/5_Batch_Prediction.py", title="Batch Prediction"),
])

pg.run()