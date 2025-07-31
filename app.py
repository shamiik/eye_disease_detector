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
# app.py

import os
import gdown
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, MultiHeadAttention, Dense
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# =============================================================================
#
#  FINAL SOLUTION: Correcting the Tensor Shape Mismatch
#
#  The error logs definitively show that the MultiHeadAttention layer is
#  receiving a 2D tensor (batch, features) but requires a 3D tensor
#  (batch, sequence, features).
#
#  This final version of the SelfAttention class fixes this by:
#  1. Adding a "sequence" dimension to the input tensor before attention.
#  2. Removing the "sequence" dimension from the output tensor after attention.
#
#  This correctly adapts the data shape and resolves the underlying error.
#
# =============================================================================

class SelfAttention(Layer):
    def __init__(self, num_heads=8, key_dim=256, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
    
    def build(self, input_shape):
        self.mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        # Input 'x' has shape (batch_size, features), e.g. (None, 1280)
        # We must add a sequence dimension to make it 3D for the MHA layer.
        x_reshaped = tf.expand_dims(x, axis=1) # Shape -> (batch_size, 1, features)
        
        # Perform attention on the correctly shaped tensor.
        attn_output = self.mha(query=x_reshaped, value=x_reshaped, key=x_reshaped)
        
        # Remove the sequence dimension to get back to a 2D tensor for the next layer.
        output_reshaped = tf.squeeze(attn_output, axis=1) # Shape -> (batch_size, features)
        
        return output_reshaped

    def get_config(self):
        config = super().get_config()
        config.update({'num_heads': self.num_heads, 'key_dim': self.key_dim})
        return config

# This function builds a new model with the same architecture as your trained one.
# This architecture is now confirmed by the error logs.
def build_model_architecture():
    # Base model, whose output is (None, 1280) as shown in the error log
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    base_model.trainable = True

    # Define the input layer
    inputs = Input(shape=(224, 224, 3))
    
    # Pass input through the base model
    x = base_model(inputs, training=False)
    
    # Reshape for the SelfAttention layer (this happens inside the layer now)
    # The name 'self_attention' must match the name in the original model
    x = SelfAttention(name='self_attention')(x)
    
    # The final prediction layer - the name 'dense' must match the original
    outputs = Dense(4, activation='softmax', name='dense')(x)

    model = Model(inputs, outputs)
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

# === Load model using the manual method ===
@st.cache_resource
def load_eye_model_manually():
    fresh_model = build_model_architecture()
    fresh_model.load_weights(model_path)
    return fresh_model

try:
    model = load_eye_model_manually()
except Exception as e:
    st.error(f"A critical error occurred while building the model or loading weights: {e}", icon="🚨")
    st.error("This may be due to an architecture mismatch. Please check the layer names ('self_attention', 'dense') and the base model in the `build_model_architecture` function.")
    st.stop()

# === Class names ===
class_names = ['Cataract', 'Glaucoma', 'Normal', 'Diabetic Retinopathy']

# === UI Header ===
# ... (rest of the UI code is unchanged) ...
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
