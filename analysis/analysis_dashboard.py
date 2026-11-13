
import streamlit as st


def analysis_dashboard(
    vision_fake_probability: float | None = None,
    faces_analyzed: int = 0,
    audio_features: dict | None = None,
) -> None:

    st.divider()
    st.header("🧠 AI Analysis")

    col1, col2, col3 = st.columns(3)

    # Vision model score
    col1.metric(
        "Vision Model Score",
        (
            f"{vision_fake_probability:.1%}"
            if vision_fake_probability is not None
            else "--"
        ),
        help=(
            "Uncalibrated fake-class score from the vision baseline. "
            "This is not a verified probability of manipulation."
        ),
    )

    # Audio status
    col2.metric(
        "Audio",
        "Available" if audio_features is not None else "Unavailable",
    )

    # Lip-sync status
    col3.metric(
        "Lip Sync Error",
        "Not added",
    )
