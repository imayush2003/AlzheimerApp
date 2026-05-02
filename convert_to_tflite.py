import tensorflow as tf

print("Loading model...")

MODEL_PATH = "models/mobilenetv2_finetuned.h5"
TFLITE_PATH = "models/mobilenetv2.tflite"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Converting to TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("✅ DONE: TFLite model created")