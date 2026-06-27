import matplotlib.pyplot as plt
import librosa
import librosa.display

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)


def create_waveform_image(audio_path, image_path="waveform.png"):
    y, sr = librosa.load(audio_path, sr=None)

    plt.figure(figsize=(7, 2.2))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Audio Waveform")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    plt.close()

    return image_path


def generate_pdf_report(
    report_path,
    reference_text,
    transcript,
    similarity,
    level,
    audio_features,
    filler_ratio,
    audio_path
):
    waveform_path = create_waveform_image(audio_path)

    doc = SimpleDocTemplate(
        report_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=16,
        leading=20,
        spaceAfter=10,
        textColor=colors.black
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        spaceAfter=10,
        textColor=colors.black
    )

    story = []

    story.append(Paragraph("Reference Concept", heading_style))
    story.append(Paragraph(reference_text, normal_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Student Transcription", heading_style))
    story.append(Paragraph(transcript, normal_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Audio Visualization", heading_style))
    story.append(Image(waveform_path, width=460, height=145))
    story.append(Spacer(1, 18))

    story.append(Paragraph("Evaluation Summary", heading_style))

    final_score = int(similarity)

    table_data = [
        ["Metric", "Value"],
        ["Semantic Similarity", str(similarity)],
        ["Filler Word Ratio", str(filler_ratio)],
        ["Pause Ratio", str(audio_features.get("pause_ratio", "N/A"))],
        ["Confidence (Energy)", str(audio_features.get("rms_energy", "N/A"))],
        ["Final Score", f"{final_score}/100"],
        ["Understanding Level", level],
    ]

    table = Table(table_data, colWidths=[260, 260])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(table)

    doc.build(story)