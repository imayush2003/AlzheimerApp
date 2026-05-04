---
title: Alzheimer AI System
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.57.0
app_file: streamlit_app/app.py
python_version: 3.10
pinned: false
---
# 🧠 Alzheimer Detection AI System

A Lightweight Deep Learning application with Explainable AI (Grad-CAM) to detect Alzheimer's Disease from MRI brain scans.

## 🌟 Features
*   **Dual Inference Modes**:
    *   **Keras (Accurate)**: Utilizes a fine-tuned MobileNetV2 model for high-accuracy predictions, including support for Grad-CAM explainability.
    *   **TFLite (Fast)**: Uses a quantized/optimized TFLite model for extremely fast and lightweight inference.
*   **Explainable AI**: Integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight the regions of the MRI scan most indicative of the diagnosis.
*   **Modern Interactive Dashboard**: Built with Streamlit, providing real-time predictions, confidence metrics, and interactive visualizations using Plotly.
*   **Comprehensive Metrics**: Displays inference time, model confidence levels, and class probabilities in dynamic charts.

## 📂 Project Structure

```
Alzheimer_Project/
│
├── Alzheimer_Dataset/                     # Dataset containing MRI images
├── models/                                # Saved weights
│   ├── mobilenetv2_finetuned.h5           # Keras H5 model
│   └── mobilenetv2.tflite                 # TFLite model
├── streamlit_app/                         # Web Application directory
│   ├── app.py                             # Main Streamlit application
│   └── runtime.txt                        # Runtime specifications
├── requirements.txt                       # Project dependencies
├── Phase1_Implementation_Notebook.ipynb   # Initial implementation and data loading
├── Phase2_MobileNetV2_GradCAM.ipynb       # Model training and Grad-CAM generation
├── Phase3.ipynb                           # Final iterations and metrics
├── convert_to_tflite.py                   # Script to convert H5 to TFLite
└── test_tflite.py                         # Script to evaluate the TFLite model
```

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Installation
Clone the repository and install the dependencies:

```bash
cd Alzheimer_Project
pip install -r requirements.txt
```

### 3. Running the App
Navigate into the `streamlit_app` directory and run the application:

```bash
cd streamlit_app
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

## 📊 Model Information
*   **Architecture**: Fine-tuned MobileNetV2
*   **Parameters**: ~2.26 Million
*   **Model Size**: ~8.63 MB
*   **Cross-Validation Accuracy**: 91.54%
*   **Classes**:
    *   Mild Dementia
    *   Moderate Dementia
    *   Non Demented
    *   Very mild Dementia

## 🛠️ Technology Stack
*   **Frontend / Dashboard**: Streamlit
*   **Deep Learning**: TensorFlow, Keras, TensorFlow Lite
*   **Computer Vision**: OpenCV, PIL
*   **Data Visualization**: Plotly, Pandas

## ⚖️ Disclaimer
This tool is for educational and research purposes only. It is not intended for clinical use or as a substitute for professional medical advice, diagnosis, or treatment.
