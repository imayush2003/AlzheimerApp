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

# =========================================
# CONFIG
# =========================================
st.set_page_config(
    page_title="Alzheimer AI Screening",
    page_icon="🧠",
    layout="wide"
)

# ---------- CUSTOM CSS (MODERN UI) ----------
st.markdown("""
<style>
.main {background: linear-gradient(135deg, #0f172a, #020617);}
.stMetric {background-color: #111827; padding: 15px; border-radius: 12px;}
.block-container {padding-top: 2rem;}
h1, h2, h3 {color: #e5e7eb;}
</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODELS
# =========================================

MODEL_PATH = "/models/mobilenetv2_finetuned.h5"
TFLITE_PATH = "../models/mobilenetv2.tflite"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.trainable = False
    return model

@st.cache_resource
def load_tflite():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    return interpreter

model = load_model()

CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

# =========================================
# PREPROCESS
# =========================================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)

# =========================================
# GRAD-CAM
# =========================================
def make_gradcam_heatmap(img_array, model):
    backbone = model.layers[1]
    gap = model.layers[2]
    dropout = model.layers[3]
    dense = model.layers[4]

    with tf.GradientTape() as tape:
        conv = backbone(img_array, training=False)
        tape.watch(conv)

        x = gap(conv)
        x = dropout(x, training=False)
        preds = dense(x)

        idx = tf.argmax(preds[0])
        loss = preds[:, idx]

    grads = tape.gradient(loss, conv)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    conv = conv[0]

    heatmap = tf.reduce_sum(conv * pooled, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

# =========================================
# OVERLAY
# =========================================
def overlay_heatmap(img, heatmap):
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Inference Mode",
        ["Keras (Accurate)", "TFLite (Fast)"]
    )

    show_gradcam = st.toggle("Enable Grad-CAM", True)

    st.divider()

    st.subheader("📊 Model Info")
    st.write("Architecture: MobileNetV2")
    st.write("Params: 2.26M")
    st.write("Model Size: 8.63 MB")
    st.write("Cross-Val Accuracy: 91.54%")

# =========================================
# MAIN UI
# =========================================
st.title("🧠 Alzheimer Detection AI System")
st.caption("Lightweight Deep Learning + Explainable AI (Grad-CAM)")

uploaded_file = st.file_uploader("Upload Brain MRI", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    img_array = preprocess_image(image)

    # -------- Inference --------
    start = time.time()

    if mode == "Keras (Accurate)":
        prediction = model.predict(img_array)[0]
    else:
        interpreter = load_tflite()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    inference_time = time.time() - start

    class_index = np.argmax(prediction)
    confidence = prediction[class_index]

    # =========================================
    # DASHBOARD METRICS
    # =========================================
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Prediction", CLASS_NAMES[class_index])
    c2.metric("Confidence", f"{confidence:.4f}")
    c3.metric("Time (ms)", f"{inference_time*1000:.2f}")
    c4.metric("Mode", mode)

    st.divider()

    # =========================================
    # TABS
    # =========================================
    tab1, tab2, tab3 = st.tabs([
        "📊 Analysis",
        "🧠 Explainability",
        "📄 Details"
    ])

    # ---------- TAB 1 ----------
    with tab1:

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, width="stretch")

        with col2:
            fig = go.Figure()

            fig.add_bar(
                x=CLASS_NAMES,
                y=prediction
            )

            fig.update_layout(
                title="Class Probabilities",
                template="plotly_dark"
            )

            st.plotly_chart(fig, width="stretch")

            # Gauge
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(confidence),
                title={'text': "Confidence"},
                gauge={'axis': {'range': [0,1]}}
            ))

            st.plotly_chart(gauge, width="stretch")

    # ---------- TAB 2 ----------
    with tab2:

        if mode == "Keras (Accurate)" and show_gradcam:

            heatmap = make_gradcam_heatmap(img_array, model)
            overlay = overlay_heatmap(image, heatmap)

            st.image(overlay, width="stretch")

        else:
            st.warning("Grad-CAM not available in TFLite mode")

    # ---------- TAB 3 ----------
    with tab3:

        st.subheader("Prediction Breakdown")

        df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": prediction
        })

        st.dataframe(df, width="stretch")

        st.subheader("Interpretation")

        if class_index == 2:
            st.success("No dementia detected.")
        elif class_index == 3:
            st.warning("Very mild cognitive decline.")
        elif class_index == 0:
            st.warning("Mild dementia detected.")
        else:
            st.error("Moderate dementia detected.")

    st.success("✅ Analysis Complete")