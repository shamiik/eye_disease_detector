import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_eye_model():
    return load_model("ensemble.h5")

model = load_eye_model()

# Define class labels
class_names = ['Cataract', 'Glaucoma', 'Normal', 'Diabetic Retinopathy']

# Page config
st.set_page_config(page_title="Eye Disease Detector", page_icon="🧠", layout="centered")

# Custom header
st.markdown("""
    <h2 style='text-align: center; color: #4B8BBE;'>👁️ Eye Disease Detector - Personalizer Mode</h2>
    <p style='text-align: center;'>Upload an eye image to detect common eye diseases using a trained deep learning model.</p>
    <hr style='border-top: 1px solid #bbb;'/>
""", unsafe_allow_html=True)

# Sidebar upload
st.sidebar.header("📤 Upload Eye Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Process image if uploaded
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("Predicting... 🔍"):
        # Preprocess image
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

    # Show result in a styled container
    st.markdown(f"""
        <div style='background-color: #e6f7ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1890ff;'>
            <h3>🧪 Prediction Result</h3>
            <p><strong>Disease:</strong> <span style='color: #d9534f;'>{predicted_class}</span></p>
            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("👈 Please upload an eye image from the sidebar to begin.")
