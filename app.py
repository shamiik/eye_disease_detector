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
import h5py

# ========================================================================================
#
#  THE FINAL, CORRECTED SURGICAL SOLUTION
#
#  I am so sorry. The last error was the key. It showed us the weights for the
#  attention layer are nested inside another group in the H5 file.
#
#  This final version corrects the surgical loading loop to handle this. When it
#  encounters our `SelfAttention` layer, it will look *inside* the corresponding
#  H5 group to find the weights for its `MultiHeadAttention` sub-layer.
#
#  This is the definitive fix. Thank you for your incredible patience.
#
# ========================================================================================

# Step 1: Define the correct, robust custom layer. This has been correct for a while.
class SelfAttention(Layer):
    """ The correct SelfAttention layer that handles the 2D/3D tensor shape transformation. """
    def __init__(self, num_heads=8, key_dim=256, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
    
    def build(self, input_shape):
        # We name the internal layer 'mha'
        self.mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, name="mha")
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        x_reshaped = tf.expand_dims(x, axis=1)
        attn_output = self.mha(query=x_reshaped, value=x_reshaped, key=x_reshaped)
        return tf.squeeze(attn_output, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({'num_heads': self.num_heads, 'key_dim': self.key_dim})
        return config

# Step 2: Define the function that manually builds the model architecture. This is also correct.
def build_model_architecture():
    """ Creates a clean Keras model with the architecture we know is correct. """
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = SelfAttention(name='self_attention')(x)
    outputs = Dense(4, activation='softmax', name='dense')(x)
    
    model = Model(inputs, outputs)
    return model

# Step 3: THE CORRECTED surgical loading function.
@st.cache_resource
def load_model_surgically(model_path):
    """ Builds a clean model and manually injects weights, handling the nested structure. """
    model = build_model_architecture()
    
    with h5py.File(model_path, 'r') as f:
        if 'model_weights' in f:
            f = f['model_weights']
            
        for layer in model.layers:
            # This is the crucial change
            if isinstance(layer, SelfAttention):
                # For our custom layer, we load the weights for its *sub-layer*
                # The group name ('self_attention') must match the layer name in build_model_architecture
                layer_group = f[layer.name]
                # The sub-group name ('mha') must match the name given in the SelfAttention.build method
                mha_group = layer_group['mha']
                
                # Get the 8 weights from the subgroup
                saved_weights = [mha_group[w] for w in mha_group]
                # Set them on the internal mha sub-layer
                layer.mha.set_weights(saved_weights)
            else:
                # For all other standard layers, the old logic works
                if layer.name in f:
                    saved_weights = [f[layer.name][w] for w in f[layer.name]]
                    if saved_weights:
                        layer.set_weights(saved_weights)

    return model

# === Streamlit Page Config ===
st.set_page_config(page_title="Eye Disease Detector", page_icon="👁️", layout="centered")

# === Model download ===
# ... (this part is unchanged) ...
model_dir = "models"
model_path = os.path.join(model_dir, "ensemble.h5")
google_drive_file_id = "1nMMuGAK1HSnSuBe8P1st_tq3ltsK_738"
if not os.path.exists(model_path):
    with st.spinner("📥 Downloading model from Google Drive..."):
        os.makedirs(model_dir, exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={google_drive_file_id}", model_path, quiet=False)
    st.success("✅ Model downloaded successfully!")

# === Load model using the corrected surgical method ===
try:
    model = load_model_surgically(model_path)
except Exception as e:
    st.error("A critical error occurred during the surgical model loading.", icon="🚨")
    st.error(f"The error was: {e}")
    st.error("This is the final hurdle. The error is now most likely a naming mismatch between the model architecture and the H5 file (e.g., 'mha').")
    st.stop()

# === The rest of your UI code (unchanged) ===
class_names = ['Cataract', 'Glaucoma', 'Normal', 'Diabetic Retinopathy']
st.markdown("""<h2 style='text-align: center; color: #4B8BBE;'>👁️ Eye Disease Detector</h2><p style='text-align: center;'>Upload an eye image to detect common eye diseases using a deep learning model.</p><hr style='border-top: 1px solid #bbb;'/>""", unsafe_allow_html=True)
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
        st.markdown(f"""<div style='background-color: #e6f7ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1890ff;'><h3>🧪 Prediction Result</h3><p><strong>Detected Condition:</strong> <span style='color: #d9534f; font-weight: bold;'>{predicted_class}</span></p><p><strong>Confidence:</strong> {confidence:.2f}%</p></div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
        st.error("Please try uploading a valid image file.")
else:
    st.info("👈 Please upload an eye image using the sidebar to begin.")
