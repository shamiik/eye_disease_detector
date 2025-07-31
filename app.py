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
# app.py - DIAGNOSTIC SCRIPT

import os
import gdown
import streamlit as st
import h5py

# ========================================================================================
#
#  FINAL STEP: DIAGNOSTIC SCRIPT TO INSPECT THE MODEL FILE
#
#  I am so sorry for the failures. We will stop guessing. This script will not try to
#  load the model. It will only open the .h5 file and print its internal structure
#  to the logs. This will tell us the EXACT names of the layers and weights.
#
#  INSTRUCTIONS:
#  1. Run this app.
#  2. The app will show an error message ("Diagnostics complete"). This is expected.
#  3. Go to the Streamlit logs (click "Manage app").
#  4. Find the section that starts with "--- H5 File Inspection ---".
#  5. Copy EVERYTHING from that line to "--- End of Inspection ---" and paste it
#     in your reply.
#
#  With that information, I can write the final, working code.
#
# ========================================================================================

def inspect_h5_file(model_path):
    """
    This function opens the H5 file and prints its structure.
    """
    st.info("🔬 Running diagnostics on the model file...")
    print("\n\n\n--- H5 File Inspection ---")
    
    try:
        with h5py.File(model_path, 'r') as f:
            # Navigate to the weights group, which is standard
            weights_group = f
            if 'model_weights' in f:
                weights_group = f['model_weights']

            print("Top-level layer/group names found:")
            top_level_names = list(weights_group.keys())
            print(top_level_names)

            # Specifically investigate the 'self_attention' layer group
            if 'self_attention' in weights_group:
                print("\n--- Inspecting the 'self_attention' group ---")
                sa_group = weights_group['self_attention']
                
                # Get the names of everything inside the 'self_attention' group
                sa_contents = list(sa_group.keys())
                print(f"Contents of 'self_attention' group: {sa_contents}")

                # If there's anything inside, it's likely a subgroup containing the real weights.
                # Let's print the contents of that subgroup.
                if sa_contents:
                    # Let's assume the first item is the subgroup we need to inspect.
                    subgroup_name_to_inspect = sa_contents[0]
                    if isinstance(sa_group[subgroup_name_to_inspect], h5py.Group):
                        subgroup = sa_group[subgroup_name_to_inspect]
                        subgroup_contents = list(subgroup.keys())
                        print(f"\n--- Inspecting subgroup '{subgroup_name_to_inspect}' ---")
                        print(f"Contents (weight names) inside '{subgroup_name_to_inspect}': {subgroup_contents}")

    except Exception as e:
        print(f"An error occurred during H5 file inspection: {e}")
        
    print("--- End of Inspection ---\n\n\n")
    st.success("Diagnostics complete. Please check the logs.")
    st.warning("Copy the log output from '--- H5 File Inspection ---' to '--- End of Inspection ---' and provide it in your next reply.")


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

# === Run the diagnostic test ===
try:
    inspect_h5_file(model_path)
    # We will stop the app here on purpose.
    st.stop()
except Exception as e:
    st.error(f"A critical error occurred while running the diagnostic script: {e}")
    st.stop()
