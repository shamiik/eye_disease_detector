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
from PIL import Image

# -------------------- CONSTANTS --------------------

GOOGLE_DRIVE_FILE_ID = "1nMMuGAK1HSnSuBe8P1st_tq3ltsK_738"
MODEL_DIR = "models"
MODEL_FILENAME = "ensemble.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
IMG_SIZE = (224, 224)


CLASS_NAMES = [
    'Eyelid', 
    'Normal', 
    'Cataract', 
    'Uveitis', 
    'Conjunctivitis' 
]
# {'Eyelid': 0, 'Normal': 1, 'Cataract': 2, 'Uveitis': 3, 'Conjunctivitis': 4}


class SelfAttention(Layer):
    """A faithful re-implementation of the SelfAttention layer from the training script."""
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
    """Creates a clean Keras model with the exact ensemble architecture."""
    inputs = Input(shape=(*img_size, 3), name="input_1")

    # VGG19 Branch
    vgg = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
    vgg.trainable = False
    x1 = vgg.output
    x1 = Lambda(lambda t: tf.reshape(t, (-1, 49, 512)))(x1)
    x1_att = SelfAttention(512, name='self_attention')(x1)
    x1 = LayerNormalization(epsilon=1e-6)(x1 + x1_att)
    x1 = GlobalAveragePooling1D()(x1)

    # MobileNetV2 Branch
    mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
    mobilenet.trainable = False
    x2 = mobilenet.output
    x2 = Lambda(lambda t: tf.reshape(t, (-1, 49, 1280)))(x2)
    x2_att = SelfAttention(1280, name='self_attention_1')(x2)
    x2 = LayerNormalization(epsilon=1e-6)(x2 + x2_att)
    x2 = GlobalAveragePooling1D()(x2)

    # Fusion and Classification
    combined = Concatenate()([x1, x2])
    x = Dense(512, activation='relu')(combined)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

# -------------------- HELPER FUNCTIONS --------------------
# This section contains functions for loading the model and making predictions.

@st.cache_resource
def load_production_model(model_path):
    """
    Builds the clean model architecture and loads the weights from the specified path.
    This "surgical" method bypasses issues with corrupted model configurations.
    """
    model = build_ensemble_model()
    model.load_weights(model_path)
    return model

def process_and_predict(uploaded_file, model):
    """
    Opens, preprocesses the uploaded image, and returns the model's prediction.
    """
    # Open the image using Pillow
    img = Image.open(uploaded_file).convert("RGB")

    # Pre-process the image for the model
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch

    # Get model prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    
    # Ensure the predicted index is valid
    if predicted_index < len(CLASS_NAMES):
        predicted_class = CLASS_NAMES[predicted_index]
    else:
        predicted_class = "Error: Invalid Prediction Index"
    
    confidence = float(predictions[0][predicted_index]) * 100
    
    return img, predicted_class, confidence

# -------------------- STREAMLIT UI --------------------

# --- Page Configuration ---
st.set_page_config(page_title="Eye Disease Detector", page_icon="👁️", layout="wide")

# --- Model Loading ---

with st.spinner("Initializing application and loading AI model..."):
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    
    try:
        model = load_production_model(MODEL_PATH)
    except Exception as e:
        st.error(f"A critical error occurred during model loading: {e}", icon="🚨")
        st.error("The application cannot start. Please contact support.")
        st.stop()

# --- UI Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    # --- Header ---
    st.markdown("## 👁️ Eye Disease Detector")
    st.markdown("Upload a eye image to detect common eye diseases using our advanced AI ensemble model.")

    # --- Image Upload ---
    st.sidebar.header("📤 Upload Retinal Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if not uploaded_file:
        st.info("👈 Please upload an eye image using the sidebar to begin analysis.")
        st.sidebar.info("Awaiting image upload...")

with col2:
    if uploaded_file is not None:
        # --- Image Processing and Prediction ---
        with st.spinner("🔬 Analyzing the image..."):
            uploaded_image, predicted_class, confidence = process_and_predict(uploaded_file, model)
        
        # --- Display Uploaded Image (Resized) ---
        st.subheader(" Uploaded Image")
        # UI FIX: Displaying the image with a fixed width for a cleaner look.
        st.image(uploaded_image, width=350)

        # --- Display Prediction Result (High-Contrast Box) ---
        st.subheader("🧪 Prediction Result")
        
        if predicted_class == "Normal":
            # UI FIX: Using a green success box for 'Normal' results.
            with st.container(border=True):
                st.success(f"**Detected Condition: {predicted_class}**")
                st.write(f"The model is **{confidence:.2f}%** confident that the eye appears to be normal.")
        else:
            # UI FIX: Using a yellow warning box for detected conditions.
            with st.container(border=True):
                st.warning(f"**Detected Condition: {predicted_class}**")
                st.write(f"The model is **{confidence:.2f}%** confident in this diagnosis.")
                st.info("**Note:** This is an AI-generated prediction. Please consult a qualified ophthalmologist for a definitive diagnosis.")

# --- Disclaimer ---
st.markdown("---")
st.warning("""
**Disclaimer:** This tool is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. 
Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.
""", icon="⚠️")
