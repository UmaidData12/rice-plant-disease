import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# ============================================================================
#                         PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Rice Plant Disease Detector",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
#                      CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    * {margin:0; padding:0; box-sizing:border-box;}

    html, body {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%) !important;
        font-family: 'Poppins', sans-serif;
        min-height: 100vh !important;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%) !important;
        padding: 20px 20px 60px 20px !important;
    }

    header[data-testid="stHeader"] {display:none !important;}

    [data-testid="stMainBlockContainer"] {
        max-width: 850px !important;
        background: linear-gradient(135deg, rgba(10, 77, 92, 0.95) 0%, rgba(13, 122, 126, 0.95) 50%, rgba(10, 90, 99, 0.95) 100%) !important;
        border-radius: 30px !important;
        padding: 40px 35px 60px 35px !important;
        box-shadow: 0 25px 70px rgba(0,0,0,0.4) !important;
        border: 1px solid rgba(77, 208, 225, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }

    .header-box {
        background:white; border-radius:30px; padding:50px 40px; margin-bottom:40px;
        box-shadow:0 20px 60px rgba(0,0,0,0.25); text-align:center;
    }

    .header-title {
        font-size:40px; font-weight:800;
        background:linear-gradient(135deg,#00897b 0%,#004d40 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        margin-bottom:12px;
    }

    .header-subtitle {font-size:16px; color:#666;}

    .uploaded-preview-title {
        font-size:18px; font-weight:700; color:#e0e0e0;
        margin-bottom:20px; margin-top:30px; text-align:center;
    }

    .result-info {text-align:center; margin-top:35px; margin-bottom:40px;}
    .result-disease-name {font-size:48px; font-weight:800; color:white; margin-bottom:15px;}
    .result-confidence-score {font-size:24px; font-weight:700; color:white;}

    .disease-info-box {
        background:linear-gradient(135deg,rgba(255,255,255,0.95) 0%,rgba(240,240,240,0.95) 100%);
        border-radius:20px; padding:30px 35px; margin:30px auto; max-width:700px;
        box-shadow:0 15px 40px rgba(0,0,0,0.2); border:2px solid rgba(0,137,123,0.3);
    }

    .disease-info-title {font-size:18px; font-weight:700; color:#00897b; margin-bottom:10px;}
    .disease-info-content {font-size:15px; color:#333; line-height:1.7;}

    </style>
""", unsafe_allow_html=True)

# ============================================================================
#                            LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    try:
        model_path = "models/cnn_best_model.h5"
        if os.path.exists(model_path):
            custom_objects = {'InputLayer': tf.keras.layers.InputLayer}
            return tf.keras.models.load_model(model_path, compile=False)
        else:
            st.error("❌ Model not found at models/cnn_best_model.h5")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


def predict_disease(img_array, model):
    class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast', 'Leaf scald', 'Sheath Blight']
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx]) * 100
    return predicted_class, confidence


def get_disease_info(disease_name):
    disease_info = {
        'Bacterial Leaf Blight': {
            'cause': 'Caused by the bacterium Xanthomonas oryzae pv. oryzae.',
            'symptoms': 'Water-soaked lesions on leaf tips and margins that turn yellow to white.',
            'management': 'Use resistant varieties, proper sanitation, avoid excess nitrogen.'
        },
        'Brown Spot': {
            'cause': 'Caused by Bipolaris oryzae fungus.',
            'symptoms': 'Circular or oval brown spots with gray or white centers.',
            'management': 'Use clean seeds, balanced fertilizer, and water management.'
        },
        'Healthy Rice Leaf': {
            'cause': 'No disease detected - plant is healthy.',
            'symptoms': 'Green, vibrant leaves.',
            'management': 'Maintain irrigation and preventive care.'
        },
        'Leaf Blast': {
            'cause': 'Caused by the fungus Magnaporthe oryzae.',
            'symptoms': 'Diamond-shaped lesions with gray-white centers.',
            'management': 'Use resistant varieties, proper spacing, and fungicides.'
        },
        'Leaf scald': {
            'cause': 'Caused by Microdochium oryzae.',
            'symptoms': 'Large oblong lesions, scalded leaf appearance.',
            'management': 'Use clean seeds, crop rotation, and balanced fertilization.'
        },
        'Sheath Blight': {
            'cause': 'Caused by the fungus Rhizoctonia solani.',
            'symptoms': 'Oval lesions on sheaths with white centers and brown margins.',
            'management': 'Avoid excess nitrogen, maintain spacing, and proper water levels.'
        }
    }
    return disease_info.get(disease_name, {
        'cause': 'Information not available.',
        'symptoms': 'Information not available.',
        'management': 'Information not available.'
    })

# ============================================================================
#                            SESSION STATE
# ============================================================================
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'predicted_class' not in st.session_state:
    st.session_state.predicted_class = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

# ============================================================================
#                            MAIN UI
# ============================================================================
st.markdown("""
    <div class="header-box">
        <div class="header-title">Rice Plant Disease Detection</div>
        <div class="header-subtitle">Upload a rice leaf image to detect diseases</div>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop file here or click Browse files\nLimit 200MB per file • JPG, JPEG, PNG, BMP",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed",
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)

# If user uploads an image
if uploaded_file is not None:
    # Use PIL Image directly for sharp display
    img_pil = Image.open(uploaded_file)
    st.session_state.original_image = img_pil

    st.markdown('<div class="uploaded-preview-title">📸 Uploaded Image Preview</div>', unsafe_allow_html=True)
    # Center the image
    col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
    with col2:
        st.image(st.session_state.original_image, caption="Uploaded Image", use_column_width=True)

    st.markdown('<br>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
    with col2:
        predict_button = st.button("PREDICT DISEASE", use_container_width=True)

    if predict_button:
        model = load_model()
        if model is not None:
            with st.spinner("🔄 Analyzing..."):
                # Resize for model prediction (128x128)
                img_resized = st.session_state.original_image.resize((128,128))
                img_array = np.array(img_resized)
                st.session_state.predicted_class, st.session_state.confidence = predict_disease(img_array, model)
                st.session_state.prediction_done = True
                st.rerun()

else:
    st.session_state.original_image = None
    st.session_state.prediction_done = False
    st.session_state.predicted_class = None
    st.session_state.confidence = None

# ============================================================================
#                            DISPLAY RESULT
# ============================================================================
if st.session_state.prediction_done and st.session_state.original_image is not None:
    pred_class = st.session_state.predicted_class
    conf = st.session_state.confidence

    st.markdown(f"""
        <div class="result-info">
            <div class="result-disease-name">{pred_class}</div>
            <div class="result-confidence-score">Confidence: {conf:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

    disease_data = get_disease_info(pred_class)
    st.markdown(f"""
        <div class="disease-info-box">
            <div class="disease-info-section">
                <div class="disease-info-title">🦠 What Causes It?</div>
                <div class="disease-info-content">{disease_data['cause']}</div>
            </div>
            <div class="disease-info-section">
                <div class="disease-info-title">🔍 Symptoms</div>
                <div class="disease-info-content">{disease_data['symptoms']}</div>
            </div>
            <div class="disease-info-section">
                <div class="disease-info-title">💊 Management</div>
                <div class="disease-info-content">{disease_data['management']}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
    with col2:
        if st.button("DETECT ANOTHER IMAGE", use_container_width=True):
            st.session_state.original_image = None
            st.session_state.prediction_done = False
            st.session_state.predicted_class = None
            st.session_state.confidence = None
            st.session_state.file_uploader_key += 1
            st.rerun()
