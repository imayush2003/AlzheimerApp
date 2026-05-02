import tensorflow as tf
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os

# ===============================
# LOAD TFLITE MODEL
# ===============================
MODEL_PATH = "models/mobilenetv2.tflite"

print("🔄 Loading TFLite model...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ Model loaded successfully")
print("📥 Input shape:", input_details[0]['shape'])

# ===============================
# PREPROCESS FUNCTION
# ===============================
def preprocess(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print("❌ Error opening image:", e)
        return None

    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32)

    img = np.expand_dims(img, axis=0)

    # MobileNetV2 preprocessing
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    return img

# ===============================
# FILE PICKER
# ===============================
print("\n📂 Please select an MRI image...")

root = tk.Tk()
root.withdraw()  # Hide main window

file_path = filedialog.askopenfilename(
    title="Select MRI Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

# ===============================
# VALIDATION
# ===============================
if not file_path:
    print("❌ No file selected. Exiting...")
    exit()

if not os.path.exists(file_path):
    print("❌ File does not exist!")
    exit()

print("📁 Selected file:", file_path)

# ===============================
# PREPROCESS IMAGE
# ===============================
img = preprocess(file_path)

if img is None:
    print("❌ Preprocessing failed.")
    exit()

# ===============================
# RUN INFERENCE
# ===============================
print("\n🚀 Running inference...")

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])[0]

# ===============================
# RESULTS
# ===============================
CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very Mild Dementia"
]

pred_index = np.argmax(output)
confidence = np.max(output)

print("\n🧠 Prediction Results")
print("-----------------------------")
print("Class:", CLASS_NAMES[pred_index])
print("Confidence:", f"{confidence:.4f}")

print("\n📊 Full Probabilities:")
for i, prob in enumerate(output):
    print(f"{CLASS_NAMES[i]}: {prob:.4f}")

print("\n✅ Done!")