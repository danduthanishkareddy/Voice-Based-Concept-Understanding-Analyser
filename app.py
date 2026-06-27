import os
import tempfile
import html
import logging

import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt

from speech_to_text import speech_to_text
from semantic_eval import semantic_similarity
from audio_utils import extract_audio_features, filler_word_ratio
from scoring_engine import evaluate_understanding
from report_generator import generate_pdf_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Application started")

st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    layout="wide"
)

st.markdown("""
<style>
header, #MainMenu, footer {
    visibility: hidden;
}

.stApp {
    background-color: #050505;
    color: white;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 980px;
}

.main-title {
    font-size: 32px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.sub-title {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 34px;
}

.divider-line {
    border-top: 1px solid #2f3540;
    margin-bottom: 32px;
}

.section-label {
    font-size: 14px;
    font-weight: 700;
    color: white;
    margin-bottom: 8px;
}

.reference-title {
    font-size: 28px;
    font-weight: 800;
    color: white;
    margin-bottom: 18px;
}

.reference-plain {
    color: white;
    font-size: 17px;
    font-weight: 700;
    line-height: 1.65;
}

.info-box {
    background-color: #0f3a5d;
    color: #0ea5e9;
    padding: 18px 20px;
    border-radius: 7px;
    margin-top: 42px;
    font-size: 15px;
    font-weight: 700;
}

.output-card {
    background-color: #111827;
    border-radius: 6px;
    margin-top: 25px;
    border: 1px solid #2f3540;
}

.success-bar {
    background-color: #14532d;
    color: #22c55e;
    padding: 12px 18px;
    font-weight: 700;
    border-radius: 6px 6px 0 0;
}

.output-body {
    padding: 24px;
}

.output-title {
    color: white;
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 14px;
}

.output-text {
    color: white;
    font-size: 14px;
    line-height: 1.7;
}

.final-score {
    color: white;
    font-size: 34px;
    font-weight: 800;
}

.final-level {
    color: #f97316;
    font-size: 26px;
    font-weight: 800;
    margin-top: 25px;
}

.metric-label {
    color: #9ca3af;
    font-size: 13px;
    font-weight: 700;
}

.metric-value {
    color: white;
    font-size: 30px;
    font-weight: 700;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #2b2d39;
    border: none;
    border-radius: 8px;
    min-height: 82px;
    padding: 18px 22px;
}

[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] small {
    color: white !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background-color: #111827 !important;
    color: white !important;
    border: 1px solid #4b5563 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

.stButton > button,
.stDownloadButton > button {
    background-color: #111827;
    color: white;
    border: 1px solid #64748b;
    border-radius: 6px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Voice-Based Concept Understanding Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated evaluation of spoken conceptual explanations using AI.</div>', unsafe_allow_html=True)
st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

reference_text = (
    "Machine Learning is a subset of artificial intelligence that allows "
    "systems to learn patterns from data and improve performance without "
    "being explicitly programmed."
)

left_col, right_col = st.columns([2.1, 1])

with left_col:
    st.markdown('<div class="section-label">Upload Student Audio (WAV)</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["wav", "mp3", "m4a", "mpeg", "aac"],
        label_visibility="visible"
    )

with right_col:
    st.markdown('<div class="reference-title">Concept Reference</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="reference-plain">{reference_text}</div>', unsafe_allow_html=True)

if not uploaded_file:
    st.markdown('<div class="info-box">Upload an audio file to begin analysis.</div>', unsafe_allow_html=True)

if uploaded_file:
    file_extension = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_audio:
        temp_audio.write(uploaded_file.read())
        audio_path = temp_audio.name

    logging.info("Audio uploaded successfully")

    try:
        st.audio(uploaded_file)

        y, sr = librosa.load(audio_path, sr=None)

        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Audio Waveform")
        ax.set_xlabel("Time")
        ax.set_ylabel("")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        if st.button("Analyze Concept Understanding"):
            with st.spinner("Processing and evaluating..."):
                transcript = speech_to_text(audio_path)
                logging.info("Speech-to-text completed")

                similarity = semantic_similarity(transcript, reference_text)
                logging.info("Semantic similarity calculated")

                audio_features = extract_audio_features(audio_path)
                filler_ratio = filler_word_ratio(transcript)
                level, color = evaluate_understanding(similarity)
                final_score = int(similarity)

                safe_transcript = html.escape(transcript)

                st.markdown('<div class="output-card">', unsafe_allow_html=True)
                st.markdown('<div class="success-bar">Analysis Completed</div>', unsafe_allow_html=True)
                st.markdown('<div class="output-body">', unsafe_allow_html=True)

                left_out, right_out = st.columns([2, 1])

                with left_out:
                    st.markdown('<div class="output-title">Transcribed Explanation</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="output-text">{safe_transcript}</div>', unsafe_allow_html=True)

                with right_out:
                    st.markdown('<div class="output-title">Final Evaluation</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Understanding Score</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="final-score">{final_score}/100</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="final-level">{level}</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)

                with m1:
                    st.markdown('<div class="metric-label">Semantic Similarity</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{similarity}</div>', unsafe_allow_html=True)

                with m2:
                    st.markdown('<div class="metric-label">Filler Word Ratio</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{filler_ratio}</div>', unsafe_allow_html=True)

                with m3:
                    st.markdown('<div class="metric-label">Confidence (Energy)</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-value">{audio_features.get("rms_energy", "N/A")}</div>',
                        unsafe_allow_html=True
                    )

                report_path = "VBCUA_Report.pdf"

                generate_pdf_report(
                    report_path,
                    reference_text,
                    transcript,
                    similarity,
                    level,
                    audio_features,
                    filler_ratio,
                    audio_path
                )

                logging.info("PDF report generated")

                with open(report_path, "rb") as file:
                    st.download_button(
                        label="Download PDF Report",
                        data=file,
                        file_name="VBCUA_Report.pdf",
                        mime="application/pdf"
                    )

                st.markdown('</div></div>', unsafe_allow_html=True)

    except Exception as e:
        logging.error(f"Application error: {e}")
        st.error("An error occurred while processing the audio.")

    finally:
        try:
            os.remove(audio_path)
        except Exception as e:
            logging.warning(f"Temporary file cleanup failed: {e}")