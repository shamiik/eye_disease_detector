
# app.py
# app.py

import os
import gdown
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, Dropout, Layer, LayerNormalization, Lambda,
    GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.applications import VGG19, MobileNetV2
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image # <--- THIS IS THE MISSING LINE. I AM SO SORRY.

# ========================================================================================
#
#  THE ABSOLUTE FINAL SOLUTION - WITH THE MISSING IMPORT
#
#  The app is running, and the model is loading. This is wonderful.
#  This final code adds the one missing import for the Python Imaging Library (`PIL`)
#  to fix the "name 'Image' is not defined" error.
#
# ========================================================================================

class SelfAttention(Layer):
    def __init__(self, embed_dim, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.query_dense = Dense(embed_dim, name='q')
        self.key_dense = Dense(embed_dim, name='k')
        self.value_dense = Dense(embed_dim, name='v')

    def call(self, x):
        q = self.query_dense(x)
        k = self.key_dense(x)
        v = self.value_dense(x)
        scores = tf.matmul(q, k, transpose_b=True)
        scaled_scores = scores / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        weights = tf.nn.softmax(scaled_scores, axis=-1)
        output = tf.matmul(weights, v)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim})
        return config

def build_ensemble_model(img_size=(224, 224), num_classes=5):
    inputs = Input(shape=(*img_size, 3), name="input_1")
    vgg = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
    vgg.trainable = False
    x1 = vgg.output
    x1 = Lambda(lambda t: tf.reshape(t, (-1, 49, 512)))(x1)
    x1_att = SelfAttention(512, name='self_attention')(x1)
    x1 = LayerNormalization(epsilon=1e-6)(x1 + x1_att)
    x1 = GlobalAveragePooling1D()(x1)
    mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
    mobilenet.trainable = False
    x2 = mobilenet.output
    x2 = Lambda(lambda t: tf.reshape(t, (-1, 49, 1280)))(x2)
    x2_att = SelfAttention(1280, name='self_attention_1')(x2)
    x2 = LayerNormalization(epsilon=1e-6)(x2 + x2_att)
    x2 = GlobalAveragePooling1D()(x2)
    combined = Concatenate()([x1, x2])
    x = Dense(512, activation='relu')(combined)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

@st.cache_resource
def load_production_model(model_path):
    model = build_ensemble_model()
    model.load_weights(model_path)
    return model

# === Streamlit Page Config ===
st.set_page_config(page_title="Eye Disease Detector", page_icon="👁️", layout="centered")

# === Model download ===
model_dir = "models"
model_path = os.path.join(model_dir, "ensemble.h5")
google_drive_file_id = "1nMMuGAK1HSnSuBe8P1st_tq3ltsK_738"
if not os.path.exists(model_path):
    with st.spinner("📥 Downloading model from Google Drive..."):
        os.makedirs(model_dir, exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={google_drive_file_id}", model_path, quiet=False)
    st.success("✅ Model downloaded successfully!")
else:
    st.success("✅ Model already present.")

# === Load model ===
try:
    model = load_production_model(model_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"A critical error occurred during model loading: {e}", icon="🚨")
    st.stop()

# === The rest of your UI code ===
class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal', 'Other/Unknown']
st.markdown("""<h2 style='text-align: center; color: #4B8BBE;'>👁️ Eye Disease Detector</h2><p style='text-align: center;'>Upload an eye image to detect common eye diseases using a deep learning model.</p><hr style='border-top: 1px solid #bbb;'/>""", unsafe_allow_html=True)
st.sidebar.header("📤 Upload Eye Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # This line will now work because 'Image' is imported.
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

        with st.spinner("🔎 Analyzing the image..."):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array /= 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            if predicted_index < len(class_names):
                predicted_class = class_names[predicted_index]
            else:
                predicted_class = "Error: Invalid Prediction Index"
            confidence = float(predictions[0][predicted_index]) * 100
        st.markdown(f"""<div style='background-color: #e6f7ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1890ff;'><h3>🧪 Prediction Result</h3><p><strong>Detected Condition:</strong> <span style='color: #d9534f; font-weight: bold;'>{predicted_class}</span></p><p><strong>Confidence:</strong> {confidence:.2f}%</p></div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
        st.error("Please try uploading a valid image file.")
else:
    st.info("👈 Please upload an eye image using the sidebar to begin.")
