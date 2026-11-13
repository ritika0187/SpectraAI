import streamlit as st


def analysis_dashboard():

    st.divider()

    st.header("🧠 AI Analysis Dashboard")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Deepfake Probability",
        "-- %",
    )

    c2.metric(
        "Confidence",
        "-- %",
    )

    c3.metric(
        "Prediction",
        "Waiting..."
    )

    st.progress(0)