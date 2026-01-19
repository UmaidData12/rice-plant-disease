import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time

# ============================================================================
#                         PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Rice Plant Disease Detector",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

APP_ACCURACY = 90.86  # Your model accuracy shown on UI

# ============================================================================
#                      CUSTOM CSS STYLING - ENHANCED DESIGN
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {margin:0; padding:0; box-sizing:border-box;}

html, body {
    background: #0a0e12 !important;
    font-family: 'Inter', sans-serif;
    min-height: 100vh !important;
}

[data-testid="stAppViewContainer"] {
    background: #0a0e12 !important;
    padding: 20px 20px 60px 20px !important;
}

header[data-testid="stHeader"] {display:none !important;}

[data-testid="stMainBlockContainer"] {
    max-width: 850px !important;
    background: rgba(15, 23, 31, 0.8) !important;
    border-radius: 30px !important;
    padding: 40px 35px 60px 35px !important;
    box-shadow: 0 25px 70px rgba(0,0,0,0.4), 0 0 100px rgba(52, 211, 153, 0.1) !important;
    border: 1px solid rgba(52, 211, 153, 0.2) !important;
    backdrop-filter: blur(10px) !important;
    position: relative;
    overflow: hidden;
}

/* Animated Background Particles */
[data-testid="stMainBlockContainer"]::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(52, 211, 153, 0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    animation: particles 20s linear infinite;
    pointer-events: none;
}

@keyframes particles {
    0% { transform: translate(0, 0); }
    100% { transform: translate(50px, 50px); }
}

/* Header Box */
.header-box {
    background: rgba(15, 23, 31, 0.9);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 30px;
    padding: 50px 40px;
    margin-bottom: 30px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.25), 0 0 60px rgba(52, 211, 153, 0.1);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.header-title {
    font-size: 42px;
    font-weight: 900;
    color: white;
    line-height: 1.2;
    margin-bottom: 10px;
    letter-spacing: -1px;
    text-shadow: 0 0 30px rgba(52, 211, 153, 0.3);
}

.hero-gradient-text {
    background: linear-gradient(135deg, #34d399 0%, #10b981 50%, #34d399 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 3s ease infinite;
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.header-subtitle {
    font-size: 16px;
    color: #94a3b8;
    font-weight: 400;
}

/* Stats Bar */
.stats-bar {
    display: flex;
    justify-content: space-around;
    gap: 15px;
    margin-bottom: 30px;
}

.stat-card {
    flex: 1;
    background: rgba(52, 211, 153, 0.05);
    border: 1px solid rgba(52, 211, 153, 0.2);
    border-radius: 16px;
    padding: 20px 15px;
    text-align: center;
}

.stat-number {
    font-size: 24px;
    font-weight: 800;
    color: #34d399;
    margin-bottom: 5px;
}

.stat-label {
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}

/* Upload Preview */
.uploaded-preview-title {
    font-size: 18px;
    font-weight: 700;
    color: #e0e0e0;
    margin-bottom: 20px;
    margin-top: 30px;
    text-align: center;
}

/* Image Container */
.image-container {
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 15px 40px rgba(0,0,0,0.3);
}

/* Result */
.result-info {
    text-align: center;
    margin-top: 25px;
    margin-bottom: 15px;
}

.result-disease-name {
    font-size: 44px;
    font-weight: 900;
    color: white;
    margin-bottom: 10px;
    text-shadow: 0 0 40px rgba(52, 211, 153, 0.4);
}

/* NEW Confidence Glow (matches your image idea) */
.confidence-score {
    font-family: 'Inter', Times-New-Roman;
    text-align: center;
    font-size: 40px;
    font-weight: 900;
    color: #a4de37;
    margin: 20px 0 15px 0;
    letter-spacing: 1px;

}

@media (max-width: 768px) {
    .header-title { font-size: 32px; }
    .result-disease-name { font-size: 32px; }
    .confidence-glow { font-size: 42px; }
    .stats-bar { flex-direction: column; }
}

/* Disease info box */
.disease-info-box {
    background: linear-gradient(135deg, rgba(15, 23, 31, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
    border-radius: 24px;
    padding: 28px 30px;
    margin: 25px auto;
    max-width: 700px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3), 0 0 40px rgba(52, 211, 153, 0.1);
    border: 2px solid rgba(52, 211, 153, 0.3);
}

.disease-info-section {
    margin-bottom: 16px;
    padding: 18px;
    background: rgba(52, 211, 153, 0.08);
    border-radius: 16px;
    border-left: 4px solid #10b981;
}

.disease-info-title {
    font-size: 16px;
    font-weight: 800;
    color: #34d399;
    margin-bottom: 10px;
}

.disease-info-content {
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.8;
    font-weight: 500;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: 2px solid rgba(52, 211, 153, 0.5) !important;
    border-radius: 12px !important;
    padding: 16px 40px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    transition: none !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 31, 0.6);
    border: 2px dashed rgba(52, 211, 153, 0.4);
    border-radius: 20px;
    padding: 30px;
}
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

            class CustomRandomFlip(tf.keras.layers.RandomFlip):
                def __init__(self, *args, **kwargs):
                    kwargs.pop("data_format", None)
                    super().__init__(*args, **kwargs)

            class CustomRandomContrast(tf.keras.layers.RandomContrast):
                def __init__(self, *args, **kwargs):
                    kwargs.pop("value_range", None)
                    super().__init__(*args, **kwargs)

            custom_objects = {
                "InputLayer": tf.keras.layers.InputLayer,
                "RandomFlip": CustomRandomFlip,
                "RandomContrast": CustomRandomContrast
            }

            return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

        st.error("❌ Model not found at models/cnn_best_model.h5")
        return None

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


def predict_disease(img_array, model):
    class_names = [
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Healthy Rice Leaf",
        "Leaf Blast",
        "Leaf scald",
        "Sheath Blight"
    ]
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)
    idx = int(np.argmax(preds[0]))
    predicted_class = class_names[idx]
    confidence = float(preds[0][idx]) * 100
    return predicted_class, confidence


def get_disease_info(disease_name):
    disease_info = {
        "Bacterial Leaf Blight": {
            "cause": "Caused by the bacterium Xanthomonas oryzae pv. oryzae.",
            "symptoms": "Water-soaked lesions on leaf tips and margins that turn yellow to white.",
            "management": "Use resistant varieties, proper sanitation, avoid excess nitrogen."
        },
        "Brown Spot": {
            "cause": "Caused by Bipolaris oryzae fungus.",
            "symptoms": "Circular or oval brown spots with gray or white centers.",
            "management": "Use clean seeds, balanced fertilizer, and water management."
        },
        "Healthy Rice Leaf": {
            "cause": "No disease detected - plant is healthy.",
            "symptoms": "Green, vibrant leaves.",
            "management": "Maintain irrigation and preventive care."
        },
        "Leaf Blast": {
            "cause": "Caused by the fungus Magnaporthe oryzae.",
            "symptoms": "Diamond-shaped lesions with gray-white centers.",
            "management": "Use resistant varieties, proper spacing, and fungicides."
        },
        "Leaf scald": {
            "cause": "Caused by Microdochium oryzae.",
            "symptoms": "Large oblong lesions, scalded leaf appearance.",
            "management": "Use clean seeds, crop rotation, and balanced fertilization."
        },
        "Sheath Blight": {
            "cause": "Caused by the fungus Rhizoctonia solani.",
            "symptoms": "Oval lesions on sheaths with white centers and brown margins.",
            "management": "Avoid excess nitrogen, maintain spacing, and proper water levels."
        }
    }
    return disease_info.get(disease_name, {
        "cause": "Information not available.",
        "symptoms": "Information not available.",
        "management": "Information not available."
    })

# ============================================================================
#                            SESSION STATE
# ============================================================================
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0
if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0.0

# ============================================================================
#                            MAIN UI
# ============================================================================
st.markdown(f"""
<div class="header-box">
    <div class="header-title">Rice Plant Disease <span class="hero-gradient-text">Detection</span></div>
    <div class="header-subtitle">AI-Powered Instant Disease Diagnosis • {APP_ACCURACY:.2f}% Accuracy</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-bar">
    <div class="stat-card">
        <div class="stat-number">6</div>
        <div class="stat-label">Diseases</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{APP_ACCURACY:.2f}%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">CNN</div>
        <div class="stat-label">Model</div>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop file here or click Browse files\nLimit 200MB per file • JPG, JPEG, PNG, BMP",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed",
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)

# ============================================================================
# UPLOAD & PREDICT
# ============================================================================
if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    st.session_state.original_image = img_pil

    st.markdown('<div class="uploaded-preview-title">📸 Uploaded Image Preview</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(st.session_state.original_image, caption="Uploaded Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
    with col2:
        predict_button = st.button("🔬 ANALYZE NOW", use_container_width=True)

    if predict_button:
        model = load_model()
        if model is not None:
            with st.spinner("🔄 Analyzing image..."):
                start_time = time.time()

                img_resized = st.session_state.original_image.resize((128, 128))
                img_array = np.array(img_resized)

                pred_class, conf = predict_disease(img_array, model)

                st.session_state.processing_time = time.time() - start_time
                st.session_state.predicted_class = pred_class
                st.session_state.confidence = float(conf)
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
    conf = (float(st.session_state.confidence))
    disease_data = get_disease_info(pred_class)

    st.markdown(f"""
    <div class="result-info">
        <div class="result-disease-name">{pred_class}</div>
    </div>
    """, unsafe_allow_html=True)

    #NEW Confidence Score display like your image
    st.markdown(
        f"""
        <div class="confidence-score">
         Confidence Score {conf:.2f}%
        </div>

        """,
        unsafe_allow_html=True
    )


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

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
    with col2:
        if st.button("🔄 ANALYZE ANOTHER IMAGE", use_container_width=True):
            st.session_state.original_image = None
            st.session_state.prediction_done = False
            st.session_state.predicted_class = None
            st.session_state.confidence = None
            st.session_state.file_uploader_key += 1
            st.rerun()
