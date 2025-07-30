import os
import gdown
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# === Streamlit Page Config ===
st.set_page_config(page_title="Eye Disease Detector", page_icon="🧠", layout="centered")

# === Model download from Google Drive ===
model_path = "ensemble.h5"
google_drive_file_id = "1nMMuGAK1HSnSuBe8P1st_tq3ltsK_738"

if not os.path.exists(model_path):
    with st.spinner("📥 Downloading model from Google Drive..."):
        gdown.download(f"https://drive.google.com/uc?id={google_drive_file_id}", model_path, quiet=False)
    st.success("✅ Model downloaded successfully!")

# === Load model without compiling (safe for inference) ===
@st.cache_resource
def load_eye_model():
    return load_model(model_path, compile=False)

model = load_eye_model()

# === Class names (update if needed) ===
class_names = ['Cataract', 'Glaucoma', 'Normal', 'Diabetic Retinopathy']

# === UI Header ===
st.markdown("""
    <h2 style='text-align: center; color: #4B8BBE;'>👁️ Eye Disease Detector - Personalizer Mode</h2>
    <p style='text-align: center;'>Upload an eye image to detect common eye diseases using a trained deep learning model.</p>
    <hr style='border-top: 1px solid #bbb;'/>
""", unsafe_allow_html=True)

# === Image Upload ===
st.sidebar.header("📤 Upload Eye Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("🔎 Making prediction..."):
        # Preprocess image
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

    # Show result
    st.markdown(f"""
        <div style='background-color: #e6f7ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1890ff;'>
            <h3>🧪 Prediction Result</h3>
            <p><strong>Disease:</strong> <span style='color: #d9534f;'>{predicted_class}</span></p>
            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("👈 Please upload an eye image from the sidebar to begin.")
