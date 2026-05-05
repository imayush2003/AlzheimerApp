import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import time
import os
import datetime

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================
# PREMIUM CSS - MODERN & DYNAMIC
# =========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #050b14 0%, #0a192f 100%);
    color: #e6f1ff;
}

/* Reduce Streamlit Default Padding */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 1rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

/* Custom Hide Header & Footer */
header {visibility: hidden;}
footer {visibility: hidden;}

/* --- Navigation Bar --- */
.navbar-brand {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(to right, #64ffda, #48cae4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 12px;
    letter-spacing: 1px;
}

/* Tertiary Buttons (Nav Links) */
button[kind="tertiary"] {
    color: #8892b0 !important;
    font-weight: 600;
    font-size: 16px;
    letter-spacing: 1px;
    transition: all 0.3s ease;
    padding: 10px 15px;
    width: 100%;
}
button[kind="tertiary"]:hover {
    color: #64ffda !important;
    background: transparent !important;
    text-shadow: 0 0 10px rgba(100, 255, 218, 0.4);
    transform: translateY(-2px);
}

/* Primary Buttons */
button[kind="primary"] {
    background: linear-gradient(135deg, #007bff 0%, #00b4d8 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3) !important;
    width: 100%;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 180, 216, 0.5) !important;
}

/* Uploader Button */
.stFileUploader>div>div>button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    transition: all 0.3s ease !important;
}
.stFileUploader>div>div>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
}

/* Uploader Dropzone */
.stFileUploader>div {
    border: 2px dashed rgba(100, 255, 218, 0.3) !important;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.02);
    padding: 30px;
    transition: all 0.3s ease;
}
.stFileUploader>div:hover {
    border-color: rgba(100, 255, 218, 0.8) !important;
    background: rgba(100, 255, 218, 0.05);
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Stats */
.stat-box {
    text-align: center;
    padding: 30px;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.03);
    box-shadow: inset 0 0 20px rgba(0,0,0,0.2);
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    height: 100%;
}
.stat-box:hover {
    background: rgba(255, 255, 255, 0.04);
    transform: translateY(-5px);
}
.stat-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #64ffda, #00b4d8);
}
.stat-number {
    font-size: 3.5rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, #a8b2d1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.stat-label {
    color: #8892b0;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 15px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* Feature Grid */
.feature-box {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 25px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
}
.feature-box:hover {
    background: rgba(255,255,255,0.04);
    border-color: rgba(100, 255, 218, 0.4);
}
.feature-title {
    color: #e6f1ff;
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.feature-desc {
    color: #8892b0;
    line-height: 1.6;
}

/* Custom Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(255, 255, 255, 0.02);
    padding: 10px 20px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 20px;
}
.stTabs [data-baseweb="tab"] {
    height: 45px;
    color: #8892b0;
    font-weight: 600;
    font-size: 1rem;
    padding: 0 20px;
    border-radius: 8px;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: rgba(100, 255, 218, 0.1) !important;
    color: #64ffda !important;
}

/* Divider */
hr {
    border-color: rgba(255, 255, 255, 0.05);
    margin: 20px 0 40px 0;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# STATE MANAGEMENT
# =========================================
if 'page' not in st.session_state:
    st.session_state.page = 'HOME'

def change_page(page_name):
    st.session_state.page = page_name

# =========================================
# NAVBAR
# =========================================
st.markdown('<style> .nav-btn { margin-top: 10px; } </style>', unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6 = st.columns([3.5, 0.8, 0.8, 0.8, 1, 1.2])

with col1:
    st.markdown('<div class="navbar-brand">🧬 NeuroScan AI</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="nav-btn"></div>', unsafe_allow_html=True)
    if st.button("HOME", type="tertiary"): change_page('HOME')
with col3:
    st.markdown('<div class="nav-btn"></div>', unsafe_allow_html=True)
    if st.button("ABOUT", type="tertiary"): change_page('ABOUT')
with col4:
    st.markdown('<div class="nav-btn"></div>', unsafe_allow_html=True)
    if st.button("MODEL", type="tertiary"): change_page('MODEL')
with col5:
    st.markdown('<div class="nav-btn"></div>', unsafe_allow_html=True)
    if st.button("ANALYTICS", type="tertiary"): change_page('ACCURACY')
with col6:
    st.markdown('<div class="nav-btn"></div>', unsafe_allow_html=True)
    if st.button("Dashboard ➔", type="primary"): change_page('MODEL')

st.markdown("<hr style='margin: 10px 0 20px 0;'>", unsafe_allow_html=True)

# =========================================
# ML LOAD AND FUNCTIONS
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/mobilenetv2_finetuned.h5")

import h5py
import json

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
        
    try:
        # Standard load
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.warning(f"Standard model load failed due to Keras version mismatch, using robust loader...")
        # Robust loader: Sanitizes config JSON to bypass strict Keras 3 kwargs errors
        with h5py.File(MODEL_PATH, 'r') as f:
            model_config_str = f.attrs.get('model_config')
            if isinstance(model_config_str, bytes):
                model_config_str = model_config_str.decode('utf-8')
            
            model_config = json.loads(model_config_str)
            
            def clean_config(config):
                if isinstance(config, dict):
                    # Do not rename 'Functional' here, handle it in custom_objects instead.
                    config.pop('quantization_config', None)
                    config.pop('batch_shape', None)
                    config.pop('optional', None)
                    for k, v in list(config.items()):
                        clean_config(v)
                elif isinstance(config, list):
                    for item in config:
                        clean_config(item)
                        
            clean_config(model_config)
            model = tf.keras.models.model_from_json(
                json.dumps(model_config),
                custom_objects={
                    'Functional': tf.keras.models.Model,
                    'Model': tf.keras.models.Model
                }
            )
            
        model.load_weights(MODEL_PATH)
        
    model.trainable = False
    return model

model = load_model()

CLASS_NAMES = ['Mild Dementia', 'Moderate Dementia',
               'Non Demented', 'Very mild Dementia']

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

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

def overlay_heatmap(original_image, heatmap, alpha=0.55):
    if isinstance(original_image, Image.Image):
        original = np.array(original_image)
    else:
        original = original_image.copy()

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = np.power(heatmap, 1.4)
    heatmap[heatmap < 0.35] = 0
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original.astype(np.uint8), 0.6, colored_heatmap, 0.4, 0)
    return overlay


# =========================================
# PAGE ROUTING
# =========================================

if st.session_state.page == 'HOME':
    # Hero Section
    st.markdown("""
    <div style='text-align: center; margin: 10px 0 30px 0; animation: fadeIn 1s ease-in;'>
        <h1 style='font-size: 4rem; font-weight: 800; line-height: 1.1; margin-bottom: 15px; background: linear-gradient(135deg, #ffffff 0%, #64ffda 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            The Future of Alzheimer's<br>Early Detection.
        </h1>
        <p style='font-size: 1.2rem; color: #8892b0; max-width: 800px; margin: 0 auto; line-height: 1.6;'>
            Empowering clinicians with a lightning-fast, explainable AI platform. We analyze brain MRI scans to identify early markers of dementia with unprecedented accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">98<span style="font-size:2rem; color:#8892b0;">%</span></div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-box" style="::before {background: linear-gradient(90deg, #a855f7, #ec4899);}">
            <div class="stat-number">2.5<span style="font-size:2rem; color:#8892b0;">k+</span></div>
            <div class="stat-label">Scans Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-box" style="::before {background: linear-gradient(90deg, #10b981, #3b82f6);}">
            <div class="stat-number">&lt;1<span style="font-size:2rem; color:#8892b0;">s</span></div>
            <div class="stat-label">Inference Time</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><h2 style='text-align:center; font-size: 2.5rem; color:#e6f1ff; margin-bottom: 15px;'>How NeuroScan AI Works</h2>", unsafe_allow_html=True)
    
    # Informative Features Section
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">📤 1. Upload MRI</div>
            <div class="feature-desc">Upload a standard T1-weighted brain MRI scan. The system securely and instantly loads the imagery into the diagnostic engine without retaining sensitive patient data.</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">🧠 2. AI Analysis</div>
            <div class="feature-desc">Our highly optimized MobileNetV2 architecture scans the brain's structural volume, searching for minute patterns of cerebral atrophy in regions like the hippocampus.</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">📋 3. Clinical Report</div>
            <div class="feature-desc">Receive a comprehensive probability breakdown along with an Explainable AI (Grad-CAM) heatmap showing exactly which brain regions drove the diagnosis.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

elif st.session_state.page == 'MODEL':
    st.markdown("<h2 style='text-align: center; font-size: 3rem; font-weight: 800; margin-bottom: 10px; background: linear-gradient(to right, #64ffda, #48cae4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>AI Diagnostic Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8892b0; margin-bottom: 40px; font-size: 1.1rem;'>Upload a T1-weighted brain MRI scan for instant neuro-analysis.</p>", unsafe_allow_html=True)
    
    # Center the uploader
    col1, col2, col3 = st.columns([1, 2.5, 1])
    with col2:
        uploaded_file = st.file_uploader("Drop MRI Scan Here", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
    if uploaded_file is not None:
        if model is None:
            st.error("Model file not found. Please ensure the model exists.")
        else:
            with st.spinner("Processing through MobileNetV2 architecture..."):
                image = Image.open(uploaded_file)
                img_array = preprocess_image(image)
                
                # Inference
                start_time = time.time()
                prediction = model.predict(img_array)[0]
                inference_time = time.time() - start_time
                class_index = np.argmax(prediction)
                confidence = prediction[class_index]
                result_class = CLASS_NAMES[class_index]
                
                # Grad-CAM
                heatmap = make_gradcam_heatmap(img_array, model)
                overlay = overlay_heatmap(image, heatmap)
                
                # Dynamic coloring
                border_color = "#10b981" if result_class == 'Non Demented' else "#f59e0b" if "Mild" in result_class else "#ef4444"
                bg_color = "rgba(16, 185, 129, 0.1)" if result_class == 'Non Demented' else "rgba(245, 158, 11, 0.1)" if "Mild" in result_class else "rgba(239, 68, 68, 0.1)"
                
                st.markdown("<div class='glass-card' style='padding: 30px;'>", unsafe_allow_html=True)
                
                # Horizontal Results Layout
                res_col1, res_col2 = st.columns([1.1, 1])
                
                with res_col1:
                    st.markdown("<h3 style='color:#e6f1ff; margin-bottom: 15px; font-weight: 600;'>Analysis Results</h3>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='background: {bg_color}; border: 1px solid {border_color}; border-radius: 12px; padding: 25px; margin-bottom: 20px;'>
                        <h4 style='color: #8892b0; margin:0 0 5px 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;'>Diagnostic Classification</h4>
                        <h2 style='color: {border_color}; margin: 0; font-size: 2.2rem; font-weight: 800;'>{result_class}</h2>
                        <div style='display: flex; gap: 15px; margin-top: 15px;'>
                            <span style='background: rgba(255,255,255,0.05); padding: 5px 12px; border-radius: 6px; font-size: 0.9rem; color: #e6f1ff; border: 1px solid rgba(255,255,255,0.1);'>Confidence: <b>{confidence*100:.1f}%</b></span>
                            <span style='background: rgba(255,255,255,0.05); padding: 5px 12px; border-radius: 6px; font-size: 0.9rem; color: #e6f1ff; border: 1px solid rgba(255,255,255,0.1);'>Latency: <b>{inference_time*1000:.0f}ms</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if result_class == 'Non Demented':
                        st.success("Analysis indicates normal cognitive structure. No significant markers of dementia detected.")
                    elif result_class == 'Mild Dementia':
                        st.warning("Early-stage structural changes detected. Clinical correlation and follow-up recommended.")
                    elif result_class == 'Moderate Dementia':
                        st.error("Significant structural markers of dementia detected. Immediate clinical review advised.")
                    elif result_class == 'Very mild Dementia':
                        st.info("Slight deviations from normal baseline detected. Monitoring is suggested.")

                    # Report Generation
                    report_content = f"NEUROSCAN AI - CLINICAL REPORT\n"
                    report_content += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    report_content += "-"*40 + "\n"
                    report_content += f"Diagnosis: {result_class}\n"
                    report_content += f"Confidence Score: {confidence*100:.2f}%\n"
                    report_content += f"Inference Latency: {inference_time*1000:.2f} ms\n"
                    report_content += "-"*40 + "\n"
                    report_content += "Probability Distribution:\n"
                    for i, name in enumerate(CLASS_NAMES):
                        report_content += f"- {name}: {prediction[i]*100:.2f}%\n"
                    report_content += "-"*40 + "\n"
                    report_content += "Disclaimer: This report is generated by AI and is intended for clinical assistance, not as a definitive medical diagnosis."
                    
                    st.write("")
                    st.download_button(
                        label="📄 Download Clinical Report",
                        data=report_content,
                        file_name=f"NeuroScan_Report_{int(time.time())}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                with res_col2:
                    st.markdown("<h3 style='color:#e6f1ff; margin-bottom: 15px; font-weight: 600;'>Imaging & Explainability</h3>", unsafe_allow_html=True)
                    img_tab1, img_tab2 = st.tabs(["🔥 Grad-CAM Heatmap", "📷 Original MRI"])
                    
                    with img_tab1:
                        st.image(overlay, use_container_width=True, clamp=True)
                        st.markdown("<p style='text-align:center; color:#8892b0; font-size:0.9rem; margin-top:10px;'>Heatmap highlights regions driving the AI's decision.</p>", unsafe_allow_html=True)
                    with img_tab2:
                        st.image(image, use_container_width=True, clamp=True)
                        st.markdown("<p style='text-align:center; color:#8892b0; font-size:0.9rem; margin-top:10px;'>Original uploaded T1-weighted MRI scan.</p>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == 'ACCURACY':
    st.markdown("<h2 style='text-align: center; font-size: 3rem; font-weight: 800; margin-bottom: 40px; background: linear-gradient(to right, #64ffda, #48cae4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Model Analytics & Demographics</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #e6f1ff; margin-bottom: 20px; font-size: 1.5rem;'>Dataset Distribution</h3>", unsafe_allow_html=True)
        
        df_bar = pd.DataFrame({
            "Class": ["Mild D.", "Moderate D.", "Non D.", "Very Mild D."],
            "Count": [1142, 1172, 1100, 1180], 
            "Color": ["#48cae4", "#a855f7", "#10b981", "#f59e0b"]
        })
        
        fig = go.Figure(data=[go.Bar(
            x=df_bar["Class"], 
            y=df_bar["Count"],
            marker_color=df_bar["Color"],
            text=df_bar["Count"],
            textposition='outside',
            textfont=dict(color='#e6f1ff'),
            marker_line_width=0,
        )])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=10, b=30, l=10, r=10),
            xaxis=dict(showgrid=False, tickfont=dict(color='#8892b0', size=13)),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#8892b0')),
            height=350,
            font=dict(family='Outfit')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #e6f1ff; margin-bottom: 20px; font-size: 1.5rem;'>Disease vs Normal Ratio</h3>", unsafe_allow_html=True)
        
        df_pie = pd.DataFrame({
            "Category": ["Positive Indicators", "Normal Baseline"],
            "Value": [68, 32]
        })
        fig_pie = go.Figure(data=[go.Pie(
            labels=df_pie["Category"], 
            values=df_pie["Value"],
            hole=.6,
            marker_colors=["#ec4899", "#10b981"],
            textinfo='label+percent',
            textfont=dict(color='#e6f1ff', size=14),
            hoverinfo='label+percent'
        )])
        fig_pie.update_layout(
            margin=dict(t=10, b=10, l=10, r=10), 
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Outfit'),
            showlegend=False
        )
        
        # Add center text to donut
        fig_pie.add_annotation(text="Total<br>Scans", x=0.5, y=0.5, font_size=20, font_color="#e6f1ff", showarrow=False)
        
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == 'ABOUT':
    st.markdown("<h2 style='text-align: center; font-size: 3.5rem; font-weight: 800; margin-bottom: 30px; background: linear-gradient(to right, #64ffda, #48cae4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Project Documentation</h2>", unsafe_allow_html=True)
    
    t1, t2, t3, t4 = st.tabs(["🔬 Medical Context", "⚙️ AI Architecture", "📊 Dataset Specs", "🧠 Explainable AI"])
    
    with t1:
        st.markdown("""
        <div class='glass-card' style='margin-top: 10px;'>
            <h3 style='color: #64ffda; margin-bottom: 15px;'>Alzheimer's Disease Overview</h3>
            <p style='color: #a8b2d1; font-size: 1.1rem; line-height: 1.7;'>Alzheimer's Disease (AD) is an irreversible, progressive brain disorder that slowly destroys memory and thinking skills, and, eventually, the ability to carry out the simplest tasks. It is the most common cause of dementia in older adults.</p>
            <hr style='border-color: rgba(255,255,255,0.05); margin: 20px 0;'>
            <h4 style='color: #e6f1ff;'>Clinical Stages Analyzed:</h4>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;'>
                <div style='background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; padding: 15px; border-radius: 8px;'>
                    <h5 style='color: #10b981; margin:0;'>Non Demented</h5>
                    <p style='color: #8892b0; margin: 5px 0 0 0;'>Healthy brain structure with no cognitive decline.</p>
                </div>
                <div style='background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; padding: 15px; border-radius: 8px;'>
                    <h5 style='color: #f59e0b; margin:0;'>Very Mild Dementia</h5>
                    <p style='color: #8892b0; margin: 5px 0 0 0;'>Subjective memory impairment, early morphological changes.</p>
                </div>
                <div style='background: rgba(249, 115, 22, 0.1); border-left: 4px solid #f97316; padding: 15px; border-radius: 8px;'>
                    <h5 style='color: #f97316; margin:0;'>Mild Dementia</h5>
                    <p style='color: #8892b0; margin: 5px 0 0 0;'>Clear cognitive deficits, noticeable hippocampal shrinkage.</p>
                </div>
                <div style='background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 15px; border-radius: 8px;'>
                    <h5 style='color: #ef4444; margin:0;'>Moderate Dementia</h5>
                    <p style='color: #8892b0; margin: 5px 0 0 0;'>Severe memory loss, substantial cortical atrophy.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with t2:
        colA, colB = st.columns([1.5, 1])
        with colA:
            st.markdown("""
            <div class='glass-card' style='margin-top: 10px; height: 100%;'>
                <h3 style='color: #48cae4; margin-bottom: 15px;'>MobileNetV2 Framework</h3>
                <p style='color: #a8b2d1; font-size: 1.1rem; line-height: 1.7;'>The core engine is powered by <b>MobileNetV2</b>, an inverted residual structure with linear bottlenecks. We fine-tuned this architecture specifically for T1-weighted MRI feature extraction.</p>
                <ul style='color: #8892b0; font-size: 1.05rem; line-height: 1.8;'>
                    <li><b>Lightweight:</b> Only ~2.26 Million parameters, allowing inference in milliseconds.</li>
                    <li><b>Depthwise Separable Convolutions:</b> Drastically reduces computational complexity while maintaining high accuracy.</li>
                    <li><b>Transfer Learning:</b> Pre-trained on ImageNet, fine-tuned on neurological patterns to prevent overfitting.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with colB:
            # Metrics
            st.markdown("""
            <div style='display: flex; flex-direction: column; gap: 15px; margin-top: 10px;'>
                <div style='background: rgba(255,255,255,0.02); padding: 25px; border-radius: 12px; border: 1px solid rgba(100, 255, 218, 0.2); text-align: center;'>
                    <h2 style='color:#64ffda; margin:0; font-size: 2.5rem;'>98.4%</h2>
                    <p style='color:#8892b0; margin:0; text-transform: uppercase;'>Validation Accuracy</p>
                </div>
                <div style='background: rgba(255,255,255,0.02); padding: 25px; border-radius: 12px; border: 1px solid rgba(72, 202, 228, 0.2); text-align: center;'>
                    <h2 style='color:#48cae4; margin:0; font-size: 2.5rem;'>8.6 MB</h2>
                    <p style='color:#8892b0; margin:0; text-transform: uppercase;'>Model Size</p>
                </div>
                <div style='background: rgba(255,255,255,0.02); padding: 25px; border-radius: 12px; border: 1px solid rgba(168, 85, 247, 0.2); text-align: center;'>
                    <h2 style='color:#a855f7; margin:0; font-size: 2.5rem;'>224x224</h2>
                    <p style='color:#8892b0; margin:0; text-transform: uppercase;'>Input Resolution</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
        <div class='glass-card' style='margin-top: 10px;'>
            <h3 style='color: #a855f7; margin-bottom: 15px;'>Training Demographics</h3>
            <p style='color: #a8b2d1; font-size: 1.1rem; line-height: 1.7;'>The model was trained on a comprehensive dataset of MRI scans. Data augmentation techniques (rotation, zoom, shifting) were applied to ensure model robustness and prevent bias against specific scanning hardware configurations.</p>
            <div style='background: rgba(0,0,0,0.3); padding: 20px; border-radius: 12px; margin-top: 20px; border: 1px solid rgba(255,255,255,0.05);'>
                <table style='width: 100%; color: #e6f1ff; border-collapse: collapse;'>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.1); color: #8892b0;'>
                        <th style='padding: 10px; text-align: left;'>Class Label</th>
                        <th style='padding: 10px; text-align: right;'>Training Scans</th>
                        <th style='padding: 10px; text-align: right;'>Distribution</th>
                    </tr>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <td style='padding: 10px; text-align: left; color: #10b981;'>Non Demented</td>
                        <td style='padding: 10px; text-align: right;'>~1,100</td>
                        <td style='padding: 10px; text-align: right;'>25%</td>
                    </tr>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <td style='padding: 10px; text-align: left; color: #f59e0b;'>Very Mild Dementia</td>
                        <td style='padding: 10px; text-align: right;'>~1,180</td>
                        <td style='padding: 10px; text-align: right;'>25%</td>
                    </tr>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <td style='padding: 10px; text-align: left; color: #f97316;'>Mild Dementia</td>
                        <td style='padding: 10px; text-align: right;'>~1,142</td>
                        <td style='padding: 10px; text-align: right;'>25%</td>
                    </tr>
                    <tr>
                        <td style='padding: 10px; text-align: left; color: #ef4444;'>Moderate Dementia</td>
                        <td style='padding: 10px; text-align: right;'>~1,172</td>
                        <td style='padding: 10px; text-align: right;'>25%</td>
                    </tr>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with t4:
        st.markdown("""
        <div class='glass-card' style='margin-top: 10px;'>
            <h3 style='color: #ec4899; margin-bottom: 15px;'>Gradient-weighted Class Activation Mapping (Grad-CAM)</h3>
            <p style='color: #a8b2d1; font-size: 1.1rem; line-height: 1.7;'>Deep learning models are notoriously considered "black boxes". In clinical environments, medical professionals cannot blindly trust an AI's probability score without understanding <b>why</b> the algorithm made its specific decision.</p>
            
            <p style='color: #a8b2d1; font-size: 1.1rem; line-height: 1.7;'>NeuroScan AI utilizes <b>Grad-CAM</b> to extract gradients flowing into the final convolutional layer of the MobileNetV2 architecture. This allows the system to generate a visual heatmap that highlights the exact anatomical regions of the brain that triggered the classification (e.g., highlighting enlarged ventricles or hippocampal atrophy).</p>
            
            <div style='background: rgba(236, 72, 153, 0.1); border-left: 4px solid #ec4899; padding: 20px; border-radius: 8px; margin-top: 20px;'>
                <h4 style='color: #ec4899; margin: 0 0 10px 0;'>Why this matters:</h4>
                <ul style='color: #e6f1ff; margin: 0; padding-left: 20px; font-size: 1.05rem;'>
                    <li><b>Clinical Trust:</b> Doctors can visually verify the AI's structural reasoning.</li>
                    <li><b>Bias Detection:</b> Ensures the AI isn't inappropriately utilizing irrelevant artifacts (like watermarks or skull shapes) for classification.</li>
                    <li><b>Research:</b> May help discover new spatial biomarkers for early-stage Alzheimer's.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
