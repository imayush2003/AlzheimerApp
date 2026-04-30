import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import time
import tempfile
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch


# =========================================
# OUTPUT FOLDER FOR PUBLICATION FIGURES
# =========================================

OUTPUT_DIR = "../outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Alzheimer AI Screening",
    page_icon="🧠",
    layout="wide"
)


# =========================================
# LOAD MODEL
# =========================================

MODEL_PATH = "../models/mobilenetv2_finetuned.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.trainable = False
    return model

model = load_model()

CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

MODEL_SIZE_MB = 8.63
CROSSVAL_ACC = 91.54


# =========================================
# PREPROCESS
# =========================================

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image).astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    return img


# =========================================
# GRAD-CAM SAFE FOR NESTED MODEL
# =========================================

def make_gradcam_heatmap(img_array, model):

    backbone = model.layers[1]
    gap_layer = model.layers[2]
    dropout_layer = model.layers[3]
    dense_layer = model.layers[4]

    with tf.GradientTape() as tape:

        conv_outputs = backbone(img_array, training=False)

        tape.watch(conv_outputs)

        x = gap_layer(conv_outputs)
        x = dropout_layer(x, training=False)
        predictions = dense_layer(x)

        class_index = tf.argmax(predictions[0])

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.nn.relu(heatmap)

    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


# =========================================
# OVERLAY
# =========================================

def overlay_heatmap(original_image, heatmap, alpha=0.45):

    if isinstance(original_image, Image.Image):
        original = np.array(original_image)
    else:
        original = original_image.copy()

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    heatmap = np.power(heatmap, 1.4)
    heatmap[heatmap < 0.35] = 0

    heatmap_uint8 = np.uint8(255 * heatmap)

    colored = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    colored = cv2.cvtColor(
        colored,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original.astype(np.uint8),
        1 - alpha,
        colored,
        alpha,
        0
    )

    return overlay


# =========================================
# SAVE FIGURES
# =========================================

def save_input_figure(image):

    path = os.path.join(OUTPUT_DIR, "input_mri.png")

    plt.figure(figsize=(4,4), dpi=300)
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()

    return path


def save_gradcam_figure(overlay):

    path = os.path.join(OUTPUT_DIR, "gradcam_overlay.png")

    plt.figure(figsize=(4,4), dpi=300)
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()

    return path


def save_probability_figure(prediction):

    path = os.path.join(OUTPUT_DIR, "probabilities.png")

    plt.figure(figsize=(6,4), dpi=300)

    plt.bar(CLASS_NAMES, prediction)

    plt.title("Class Probability Distribution")
    plt.ylabel("Probability")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(path, dpi=300)
    plt.close()

    return path


# =========================================
# PDF REPORT
# =========================================

def generate_pdf_report(pred_class, confidence, inference_time, prediction):

    buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    doc = SimpleDocTemplate(buffer.name)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "<b>Alzheimer AI Screening Report</b>",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 0.3 * inch))

    elements.append(
        Paragraph(
            f"Prediction: {pred_class}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Confidence: {confidence:.4f}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Inference Time: {inference_time*1000:.2f} ms",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 0.3 * inch))

    data = [["Class", "Probability"]]

    for i in range(len(CLASS_NAMES)):
        data.append([
            CLASS_NAMES[i],
            f"{prediction[i]:.4f}"
        ])

    table = Table(data)

    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))

    elements.append(table)

    doc.build(elements)

    return buffer.name


# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.header("Model Info")

    st.write("Architecture: MobileNetV2")
    st.write(f"Model Size: {MODEL_SIZE_MB} MB")
    st.write(f"5-Fold CV: {CROSSVAL_ACC}%")

    show_gradcam = st.toggle("Enable Grad-CAM", True)


# =========================================
# UI
# =========================================

st.title("🧠 Lightweight Alzheimer Detection")


uploaded_file = st.file_uploader(
    "Upload Brain MRI",
    type=["jpg", "png", "jpeg"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    img_array = preprocess_image(image)

    col1, col2 = st.columns(2)

    # ---------- prediction ----------

    start = time.time()

    prediction = model.predict(img_array)[0]

    inference_time = time.time() - start

    class_index = np.argmax(prediction)

    confidence = prediction[class_index]


    # ---------- left ----------

    with col1:

        st.image(image, width="stretch")

        st.success(
            f"Prediction: {CLASS_NAMES[class_index]}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.4f}"
        )

        st.metric(
            "Inference Time",
            f"{inference_time*1000:.2f} ms"
        )

        if confidence > 0.85:
            st.success("High confidence")
        elif confidence > 0.6:
            st.warning("Moderate confidence")
        else:
            st.error("Low confidence")


    # ---------- right ----------

    with col2:

        df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": prediction
        })

        fig = go.Figure()

        fig.add_bar(
            x=df["Class"],
            y=df["Probability"]
        )

        fig.update_layout(
            template="plotly_dark",
            height=400
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        for i, p in enumerate(prediction):

            st.write(
                f"{CLASS_NAMES[i]}: {p:.3f}"
            )


        # GradCAM

        if show_gradcam:

            heatmap = make_gradcam_heatmap(
                img_array,
                model
            )

            overlay = overlay_heatmap(
                image,
                heatmap
            )

            st.image(
                overlay,
                width="stretch"
            )


    # ---------- save figures ----------

    input_path = save_input_figure(image)

    overlay_path = save_gradcam_figure(overlay)

    prob_path = save_probability_figure(prediction)

    st.success("Figures saved to outputs/figures")


    # ---------- PDF ----------

    pdf_path = generate_pdf_report(
        CLASS_NAMES[class_index],
        confidence,
        inference_time,
        prediction
    )

    with open(pdf_path, "rb") as f:

        st.download_button(
            "Download PDF Report",
            f,
            file_name="Alzheimer_Report.pdf",
            mime="application/pdf"
        )