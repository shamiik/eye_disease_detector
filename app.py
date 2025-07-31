import os
import gdown
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import SelfAttention
import numpy as np
from PIL import Image

# === Streamlit Page Config ===
st.set_page_config(page_title="Eye Disease Detector", page_icon="👁️", layout="centered")

# === Model download ===
# It's good practice to place large files that don't change often in a separate directory
model_dir = "models"
model_path = os.path.join(model_dir, "ensemble.h5")
google_drive_file_id = "1nMMuGAK1HSnSuBe8P1st_tq3ltsK_738"

if not os.path.exists(model_path):
    with st.spinner("📥 Downloading model from Google Drive... This may take a moment."):
        os.makedirs(model_dir, exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={google_drive_file_id}", model_path, quiet=False)
    st.success("✅ Model downloaded successfully!")

# === Load model with SelfAttention ===
@st.cache_resource
def load_eye_model():
    # Load the model with the custom object pointing to the Keras implementation
    return load_model(model_path, custom_objects={"SelfAttention": SelfAttention})

model = load_eye_model()


# === Class names ===
class_names = ['Cataract', 'Glaucoma', 'Normal', 'Diabetic Retinopathy']

# === UI Header ===
st.markdown("""
    <h2 style='text-align: center; color: #4B8BBE;'>👁️ Eye Disease Detector</h2>
    <p style='text-align: center;'>Upload an eye image to detect common eye diseases using a deep learning model.</p>
    <hr style='border-top: 1px solid #bbb;'/>
""", unsafe_allow_html=True)

# === Image Upload ===
st.sidebar.header("📤 Upload Eye Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

        with st.spinner("🔎 Analyzing the image..."):
            # Preprocess image
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Create a batch

            # Predict
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[0][predicted_index]) * 100

        # Show result
        st.markdown(f"""
            <div style='background-color: #e6f7ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1890ff;'>
                <h3>🧪 Prediction Result</h3>
                <p><strong>Detected Condition:</strong> <span style='color: #d9534f; font-weight: bold;'>{predicted_class}</span></p>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please try uploading a different image file.")

else:
    st.info("👈 Please upload an eye image using the sidebar to begin.")
