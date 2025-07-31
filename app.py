# app.py

# import os
# import gdown
# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image

# # --- IMPORTANT: Revert to the original tensorflow_addons import ---
# import tensorflow_addons as tfa

# # === Streamlit Page Config ===
# st.set_page_config(page_title="Eye Disease Detector", page_icon="👁️", layout="centered")

# # === Model download ===
# model_dir = "models"
# model_path = os.path.join(model_dir, "ensemble.h5")
# google_drive_file_id = "1nMMuGAK1HSnSuBe8P1st_tq3ltsK_738"

# if not os.path.exists(model_path):
#     with st.spinner("📥 Downloading model from Google Drive..."):
#         os.makedirs(model_dir, exist_ok=True)
#         gdown.download(f"https://drive.google.com/uc?id={google_drive_file_id}", model_path, quiet=False)
#     st.success("✅ Model downloaded successfully!")

# # === Load model with the original SelfAttention layer ===
# @st.cache_resource
# def load_eye_model():
#     # Now, we load the model by telling it exactly where to find the original SelfAttention layer
#     return load_model(
#         model_path,
#         custom_objects={"SelfAttention": tfa.layers.SelfAttention}
#     )

# try:
#     model = load_eye_model()
# except Exception as e:
#     st.error(f"Error loading the model: {e}", icon="🚨")
#     st.stop()

# # === Class names ===
# class_names = ['Cataract', 'Glaucoma', 'Normal', 'Diabetic Retinopathy']

# # === UI Header ===
# st.markdown("""
#     <h2 style='text-align: center; color: #4B8BBE;'>👁️ Eye Disease Detector</h2>
#     <p style='text-align: center;'>Upload an eye image to detect common eye diseases using a deep learning model.</p>
#     <hr style='border-top: 1px solid #bbb;'/>
# """, unsafe_allow_html=True)

# # === Image Upload ===
# st.sidebar.header("📤 Upload Eye Image")
# uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     try:
#         img = Image.open(uploaded_file).convert("RGB")
#         st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

#         with st.spinner("🔎 Analyzing the image..."):
#             img_resized = img.resize((224, 224))
#             img_array = image.img_to_array(img_resized)
#             img_array /= 255.0
#             img_array = np.expand_dims(img_array, axis=0)

#             predictions = model.predict(img_array)
#             predicted_index = np.argmax(predictions[0])
#             predicted_class = class_names[predicted_index]
#             confidence = float(predictions[0][predicted_index]) * 100

#         st.markdown(f"""
#             <div style='background-color: #e6f7ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1890ff;'>
#                 <h3>🧪 Prediction Result</h3>
#                 <p><strong>Detected Condition:</strong> <span style='color: #d9534f; font-weight: bold;'>{predicted_class}</span></p>
#                 <p><strong>Confidence:</strong> {confidence:.2f}%</p>
#             </div>
#         """, unsafe_allow_html=True)

#     except Exception as e:
#         st.error(f"An error occurred during image processing: {e}")
#         st.error("Please try uploading a valid image file.")

# else:
#     st.info("👈 Please upload an eye image using the sidebar to begin.")


# app.py

import os
import gdown
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# =============================================================================
#
#  FINAL SOLUTION v4: Explicitly Accepting the Problematic Argument
#
#  The error 'Unrecognized keyword arguments: ['batch_shape']' persists.
#  This final version directly addresses the error by adding 'batch_shape=None'
#  to the constructor's signature. This forces the layer to formally
#  accept the argument during model loading, even though we will not use it,
#  resolving the conflict with the Keras `load_model` function.
#
# =============================================================================
class SelfAttention(Layer):
    # The key fix: We explicitly add batch_shape to the signature.
    def __init__(self, num_heads=8, key_dim=256, batch_shape=None, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.layernorm = LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
        })
        return config


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

# === Load model with our custom SelfAttention layer ===
@st.cache_resource
def load_eye_model():
    return load_model(
        model_path,
        custom_objects={"SelfAttention": SelfAttention}
    )

try:
    model = load_eye_model()
except Exception as e:
    st.error(f"Error loading the model: {e}", icon="🚨")
    st.stop()

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
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

        with st.spinner("🔎 Analyzing the image..."):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array /= 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[0][predicted_index]) * 100

        st.markdown(f"""
            <div style='background-color: #e6f7ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1890ff;'>
                <h3>🧪 Prediction Result</h3>
                <p><strong>Detected Condition:</strong> <span style='color: #d9534f; font-weight: bold;'>{predicted_class}</span></p>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
        st.error("Please try uploading a valid image file.")

else:
    st.info("👈 Please upload an eye image using the sidebar to begin.")
