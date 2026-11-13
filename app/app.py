
import sys
import os
import tempfile
from pathlib import Path

# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa.display

from styles import CUSTOM_CSS
from components import hero

from utils.video_info import get_video_info

from preprocessing.video.frame_extractor import FrameExtractor
from preprocessing.face.face_detector import FaceDetector
from preprocessing.face.face_cropper import FaceCropper

from preprocessing.audio.audio_processor import AudioProcessor

from models.vision.deepfake_detector import (
    score_faces,
    summarize_scores,
)

from analysis.analysis_dashboard import analysis_dashboard


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="SpectraAI",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------

hero()


# ---------------------------------------------------------
# VIDEO UPLOAD
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a video for analysis",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_file is None:

    st.info(
        "Upload a video to begin multimodal deepfake analysis."
    )

    st.stop()


# ---------------------------------------------------------
# UPLOAD SIZE VALIDATION
# ---------------------------------------------------------

MAX_UPLOAD_SIZE_MB = 200
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:

    st.error(
        f"File is too large. Please upload a video smaller than "
        f"{MAX_UPLOAD_SIZE_MB} MB."
    )

    st.stop()


# ---------------------------------------------------------
# SAVE UPLOADED VIDEO TEMPORARILY
# ---------------------------------------------------------

suffix = Path(uploaded_file.name).suffix.lower()
temp_path = None

try:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:

        temp_path = temp_file.name
        temp_file.write(uploaded_file.getbuffer())

except OSError as error:

    # Remove partially created temporary file, if present.
    if temp_path and os.path.exists(temp_path):

        try:
            os.remove(temp_path)

        except OSError:
            pass

    st.error(
        f"Could not save the uploaded video temporarily: {error}"
    )

    st.stop()


# ---------------------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------------------

try:

    # =====================================================
    # 1. VIDEO INFORMATION
    # =====================================================

    info = get_video_info(temp_path)


    # =====================================================
    # 2. AUDIO PROCESSING
    # =====================================================

    audio_features = None

    audio_processor = AudioProcessor()

    try:

        with st.spinner("Extracting and analysing audio..."):

            audio_features = audio_processor.process(
                temp_path
            )

    except Exception as error:

        st.warning(
            f"Audio could not be processed: {error}"
        )


    # =====================================================
    # 3. VIDEO PREVIEW
    # =====================================================

    st.divider()

    st.subheader("🎬 Video Preview")

    st.video(uploaded_file)


    # =====================================================
    # 4. VIDEO INFORMATION
    # =====================================================

    st.subheader("📊 Video Information")

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)

    info_col1.metric(
        "Frames",
        info.get("frames", "--")
    )

    info_col2.metric(
        "FPS",
        (
            f"{info.get('fps', 0):.2f}"
            if info.get("fps") is not None
            else "--"
        )
    )

    info_col3.metric(
        "Duration",
        (
            f"{info.get('duration', 0):.2f}s"
            if info.get("duration") is not None
            else "--"
        )
    )

    info_col4.metric(
        "Resolution",
        str(info.get("resolution", "--"))
    )


    # =====================================================
    # 5. AUDIO STATUS
    # =====================================================

    st.divider()

    st.subheader("🎧 Audio Track")

    if audio_features is not None:

        audio_col1, audio_col2 = st.columns(2)

        audio_col1.metric(
            "Audio",
            "Available"
        )

        audio_col2.metric(
            "Sample Rate",
            f"{audio_features['sample_rate']} Hz"
        )

    else:

        st.warning(
            "Audio analysis is unavailable for this video."
        )


    # =====================================================
    # 6. FRAME EXTRACTION
    # =====================================================

    st.divider()

    st.subheader("🖼️ Frame Extraction")

    extractor = FrameExtractor(
        sample_count=12
    )

    with st.spinner("Extracting video frames..."):

        frames = extractor.extract_frames(
            temp_path
        )

    st.success(
        f"Extracted {len(frames)} frames."
    )


    # =====================================================
    # 7. EXTRACTED FRAME GALLERY
    # =====================================================

    st.subheader("🎞️ Extracted Frames")

    if frames:

        for row_start in range(0, len(frames), 4):

            cols = st.columns(4)

            row_frames = frames[
                row_start:row_start + 4
            ]

            for offset, frame_data in enumerate(row_frames):

                frame = np.asarray(
                    frame_data["image"]
                )

                with cols[offset]:

                    st.image(
                        frame,
                        caption=(
                            f"Sample {row_start + offset + 1}"
                            f" • Video Frame {frame_data['index']}"
                        ),
                        channels="RGB",
                        width="stretch",
                    )

    else:

        st.warning(
            "No frames were extracted."
        )


    # =====================================================
    # 8. FACE DETECTION
    # =====================================================

    st.divider()

    st.header("👤 Face Detection")

    face_detector = FaceDetector()
    face_cropper = FaceCropper()

    all_faces = []
    face_results = []
    face_metadata = []

    with st.spinner("Detecting faces in video frames..."):

        for frame_data in frames:

            frame = np.asarray(frame_data["image"])

            annotated_frame, detected_faces = (
                face_detector.draw(frame)
            )

            cropped_faces = face_cropper.crop(
                frame,
                detected_faces
            )

            all_faces.extend(cropped_faces)

            # Preserve the original frame index for every crop.
            for crop_number, face in enumerate(cropped_faces, start=1):

                face_metadata.append({
                    "frame_index": frame_data["index"],
                    "face_number": crop_number,
                })

            face_results.append({
                "index": frame_data["index"],
                "image": annotated_frame,
                "count": len(detected_faces),
            })

    if all_faces:

        st.success(
            f"Detected {len(all_faces)} face crops."
        )

    else:

        st.warning(
            "No faces detected in the sampled frames."
        )


    # =====================================================
    # 9. DISPLAY ANNOTATED FRAMES
    # =====================================================

    st.subheader("🟩 Detected Faces")

    if face_results:

        for row_start in range(0, len(face_results), 4):

            cols = st.columns(4)

            row_results = face_results[
                row_start:row_start + 4
            ]

            for offset, result in enumerate(row_results):

                with cols[offset]:

                    st.image(
                        result["image"],
                        caption=(
                            f"Video Frame {result['index']} • "
                            f"{result['count']} face(s) detected"
                        ),
                        channels="RGB",
                        width="stretch",
                    )

    else:

        st.info(
            "No annotated frames available."
        )


    # =====================================================
    # 10. FACE CROP GALLERY
    # =====================================================

    st.divider()

    st.header("🧑‍💻 Face Crops")

    if all_faces:

        for row_start in range(0, len(all_faces), 4):

            cols = st.columns(4)

            row_faces = all_faces[
                row_start:row_start + 4
            ]

            for offset, face in enumerate(row_faces):

                with cols[offset]:

                    st.image(
                        np.asarray(face),
                        caption=(
                            f"Face Crop {row_start + offset + 1}"
                        ),
                        channels="RGB",
                        width="stretch",
                    )

    else:

        st.info(
            "No face crops available."
        )


    # =====================================================
    # 11. VISION DEEPFAKE ANALYSIS
    # =====================================================

    vision_fake_probability = None
    scores = []

    if all_faces:

        st.divider()

        st.subheader("👁️ Vision Deepfake Analysis")

        try:

            with st.spinner("Running vision analysis..."):

                scores = score_faces(all_faces)

                summary = summarize_scores(scores)

            vision_fake_probability = summary.get(
                "fake_probability"
            )

            if scores and vision_fake_probability is not None:

                display_score = float(
                    np.clip(
                        vision_fake_probability,
                        0.0,
                        1.0
                    )
                )

                # -------------------------------------------------
                # OVERALL SCORE
                # -------------------------------------------------

                st.markdown("### Overall Vision Model Score")

                score_col, gauge_col = st.columns([1, 2])

                with score_col:

                    st.metric(
                        "Average Fake-Class Score",
                        f"{display_score:.1%}"
                    )

                with gauge_col:

                    st.markdown("**Synthetic Score Indicator**")

                    st.progress(display_score)

                st.caption(
                    "Uncalibrated fake-class model score. "
                    "This is not a verified probability of manipulation."
                )


                # -------------------------------------------------
                # SCORE DISTRIBUTION
                # -------------------------------------------------

                st.divider()

                st.markdown("### 📊 Face Score Distribution")

                individual_scores = [
                    float(item["fake_probability"])
                    for item in scores
                ]

                min_score = min(individual_scores)
                max_score = max(individual_scores)

                median_score = float(
                    np.median(individual_scores)
                )

                dist_col1, dist_col2, dist_col3 = st.columns(3)

                dist_col1.metric(
                    "Minimum Score",
                    f"{min_score:.1%}"
                )

                dist_col2.metric(
                    "Maximum Score",
                    f"{max_score:.1%}"
                )

                dist_col3.metric(
                    "Median Score",
                    f"{median_score:.1%}"
                )


                # -------------------------------------------------
                # INDIVIDUAL FACE SCORES
                # -------------------------------------------------

                st.divider()

                st.markdown("### 🔍 Individual Face Predictions")

                score_rows = []

                for index, result in enumerate(scores):

                    metadata = (
                        face_metadata[index]
                        if index < len(face_metadata)
                        else {}
                    )

                    score_rows.append({

                        "Face Crop": index + 1,

                        "Video Frame": metadata.get(
                            "frame_index",
                            "Unknown"
                        ),

                        "Prediction": result["prediction"],

                        "Fake-Class Score": (
                            f"{float(result['fake_probability']):.2%}"
                        ),

                    })

                st.dataframe(
                    score_rows,
                    width="stretch",
                    hide_index=True,
                )


                # -------------------------------------------------
                # SCORE BAR CHART
                # -------------------------------------------------

                st.markdown("### 📈 Fake-Class Score per Face")

                fig, ax = plt.subplots(
                    figsize=(12, 4)
                )

                face_numbers = list(
                    range(1, len(individual_scores) + 1)
                )

                ax.bar(
                    face_numbers,
                    individual_scores
                )

                ax.axhline(
                    y=0.5,
                    linestyle="--",
                    label="50% reference line"
                )

                ax.set_ylim(0, 1)

                ax.set_xlabel("Face Crop")
                ax.set_ylabel("Fake-Class Score")

                ax.set_title(
                    "Individual Face-Crop Scores"
                )

                ax.set_xticks(face_numbers)

                ax.legend()

                ax.grid(
                    axis="y",
                    alpha=0.25
                )

                fig.tight_layout()

                st.pyplot(
                    fig,
                    width="stretch"
                )

                plt.close(fig)

                st.info(
                    "The 50% line is only a reference point, not a "
                    "validated decision threshold. A low score does "
                    "not establish that a video is authentic."
                )

            else:

                st.warning(
                    "The vision model did not return valid face scores."
                )

        except Exception as error:

            st.warning(
                f"Vision analysis unavailable: {error}"
            )


    # =====================================================
    # 12. AI ANALYSIS DASHBOARD
    # =====================================================

    analysis_dashboard(
        vision_fake_probability=vision_fake_probability,
        faces_analyzed=len(all_faces),
        audio_features=audio_features,
    )


    # =====================================================
    # 13. AUDIO VISUALIZATION
    # =====================================================

    if audio_features is not None:

        st.divider()

        st.header("🎧 Audio Analysis")

        st.success(
            "Audio extracted successfully."
        )

        sample_rate = audio_features["sample_rate"]

        audio_duration = audio_features["duration"]

        waveform = np.asarray(
            audio_features["waveform"]
        )

        mel_spectrogram = np.asarray(
            audio_features["mel_spectrogram"]
        )


        # -------------------------------------------------
        # AUDIO INFORMATION
        # -------------------------------------------------

        audio_info_col1, audio_info_col2 = st.columns(2)

        audio_info_col1.metric(
            "Sample Rate",
            f"{sample_rate} Hz"
        )

        audio_info_col2.metric(
            "Duration",
            f"{audio_duration:.2f} seconds"
        )


        # -------------------------------------------------
        # WAVEFORM
        # -------------------------------------------------

        st.subheader("🌊 Waveform")

        waveform_time = np.linspace(
            0,
            audio_duration,
            len(waveform)
        )

        fig_waveform, ax_waveform = plt.subplots(
            figsize=(14, 4)
        )

        ax_waveform.plot(
            waveform_time,
            waveform
        )

        ax_waveform.set_title(
            "Audio Waveform"
        )

        ax_waveform.set_xlabel(
            "Time (seconds)"
        )

        ax_waveform.set_ylabel(
            "Amplitude"
        )

        ax_waveform.grid(
            alpha=0.25
        )

        fig_waveform.tight_layout()

        st.pyplot(
            fig_waveform,
            width="stretch"
        )

        plt.close(fig_waveform)


        # -------------------------------------------------
        # LOG-MEL SPECTROGRAM
        # -------------------------------------------------

        st.subheader(
            "🎵 Log-Mel Spectrogram"
        )

        fig_mel, ax_mel = plt.subplots(
            figsize=(14, 7)
        )

        img = librosa.display.specshow(
            mel_spectrogram,
            sr=sample_rate,
            x_axis="time",
            y_axis="mel",
            cmap="magma",
            ax=ax_mel,
        )

        ax_mel.set_title(
            "Log-Mel Spectrogram"
        )

        ax_mel.set_xlabel(
            "Time (seconds)"
        )

        ax_mel.set_ylabel(
            "Frequency (Hz)"
        )

        fig_mel.colorbar(
            img,
            ax=ax_mel,
            format="%+2.0f dB",
            label="Power (dB)"
        )

        fig_mel.tight_layout()

        st.pyplot(
            fig_mel,
            width="stretch"
        )

        plt.close(fig_mel)


    # =====================================================
    # 14. CURRENT ANALYSIS CAPABILITIES
    # =====================================================

    st.divider()

    st.subheader(
        "🔬 Current Analysis Capabilities"
    )

    cap_col1, cap_col2, cap_col3 = st.columns(3)

    with cap_col1:

        st.markdown(
            """
            **👁️ Vision**

            - Frame extraction
            - Face detection
            - Face cropping
            - Vision deepfake baseline
            """
        )

    with cap_col2:

        st.markdown(
            """
            **🎧 Audio**

            - Audio extraction
            - WAV processing
            - Waveform analysis
            - Log-Mel spectrogram
            """
        )

    with cap_col3:

        st.markdown(
            """
            **🧠 Multimodal**

            - Vision evidence
            - Audio evidence
            - Lip-sync analysis
            - Multimodal fusion

            *Additional modules are under development.*
            """
        )


# ---------------------------------------------------------
# MAIN PROCESSING ERROR HANDLING
# ---------------------------------------------------------

except Exception as error:

    st.error(
        "Video analysis could not be completed. "
        "The file may be corrupted, unsupported, or contain "
        "data that could not be processed."
    )

    st.caption(
        f"Technical details: {error}"
    )


# ---------------------------------------------------------
# TEMPORARY FILE CLEANUP
# ---------------------------------------------------------

finally:

    if temp_path:

        try:

            if os.path.exists(temp_path):
                os.remove(temp_path)

        except OSError as cleanup_error:

            st.warning(
                f"Temporary-file cleanup failed: {cleanup_error}"
            )
