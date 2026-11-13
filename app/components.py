import streamlit as st


def hero():

    st.markdown(
        """
        <div class='main-title'>
        🎭 SpectraAI
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='subtitle'>
        Multimodal Deepfake Detection & Restoration
        </div>
        """,
        unsafe_allow_html=True,
    )


def metrics(
    frame_count: int | None = None,
    face_count: int | None = None,
    has_audio: bool | None = None,
    prediction: str = "Model pending",
) -> None:
    """Render the four summary metrics shown below the SpectraAI title."""

    frames_value = str(frame_count) if frame_count is not None else "--"
    faces_value = str(face_count) if face_count is not None else "--"

    if has_audio is None:
        audio_value = "--"
    elif has_audio:
        audio_value = "Detected"
    else:
        audio_value = "Not detected"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Frames", frames_value)

    with c2:
        st.metric("Faces", faces_value)

    with c3:
        st.metric("Audio", audio_value)

    with c4:
        st.metric("Prediction", prediction)
