import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from PIL import Image
import time


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Arabic Sign Language AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
    
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .prediction-title {
        color: #8892b0;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }
    .prediction-letter {
        color: #00ff88;
        font-family: 'Roboto', 'Orbitron', 'Arial', sans-serif;
        font-size: 6rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    }
    .prediction-sub {
        color: #8892b0;
        font-size: 1rem;
        margin-top: -10px;
        letter-spacing: 1px;
    }
    .confidence-bar-container {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin-top: 20px;
        height: 10px;
    }
    .confidence-bar-fill {
        height: 10px;
        border-radius: 10px;
        background: linear-gradient(90deg, #00ff88, #60efff);
        transition: width 0.3s ease-in-out;
    }
    .info-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(5px);
        border-radius: 15px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #00ff88;
        margin-top: 40px;
    }
    .info-header {
        font-family: 'Orbitron', sans-serif;
        color: #00ff88;
        font-size: 1.4rem;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 30px;
    }
    .info-text {
        color: #ccd6f6;
        line-height: 1.8;
    }
    .arabic-text {
        direction: rtl;
        text-align: right;
        font-family: 'Roboto', sans-serif;
    }
    .sidebar-logo {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_assets():
    model = joblib.load("models/hand_sign_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_assets()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# --- MEDIAPIPE TASKS SETUP ---
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

@st.cache_resource
def get_hand_landmarker():
    # Ensure the model file is present
    model_path = "hand_landmarker.task"
    
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return vision.HandLandmarker.create_from_options(options)

landmarker = get_hand_landmarker()
# --- ARABIC LABEL MAPPING ---
ARABIC_LABELS = {
    "Ain": "ع", "Al": "ال", "Alef": "أ", "Beh": "ب", "Dad": "ض",
    "Dal": "د", "Feh": "ف", "Ghain": "غ", "Hah": "ح", "Heh": "هـ",
    "Jeem": "ج", "Kaf": "ك", "Khah": "خ", "Laa": "لا", "Lam": "ل",
    "Meem": "م", "Noon": "ن", "Qaf": "ق", "Reh": "ر", "Sad": "ص",
    "Seen": "س", "Sheen": "ش", "Tah": "ط", "Teh": "ت", "Teh_Marbuta": "ة",
    "Theh": "ث", "Waw": "و", "Yeh": "ي", "Zah": "ظ", "Zain": "ز", "thal": "ذ"
}

# --- HAND DRAWING UTILS ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def draw_landmarks(image, hand_landmarks_list):
    h, w, _ = image.shape
    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        start_pt = hand_landmarks_list[start_idx]
        end_pt = hand_landmarks_list[end_idx]
        cv2.line(image, 
                 (int(start_pt.x * w), int(start_pt.y * h)), 
                 (int(end_pt.x * w), int(end_pt.y * h)), 
                 (0, 255, 136), 2)
    # Draw points
    for lm in hand_landmarks_list:
        cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 5, (255, 255, 255), -1)

def extract_landmarks(hand_landmarks_list):
    # hand_landmarks_list is a list of NormalizedLandmark objects
    landmarks = []
    for lm in hand_landmarks_list:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    # Create feature names to match training (x0, y0, z0, ..., x20, y20, z20)
    feature_names = []
    for i in range(21):
        feature_names.extend([f'x{i}', f'y{i}', f'z{i}'])
        
    import pandas as pd
    return pd.DataFrame([landmarks], columns=feature_names)

# --- SIDEBAR ---
with st.sidebar:
    logo_path = os.path.join("logo", "coai_logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=300)
    
    st.title("Settings")
    # Using key to preserve state across reruns better
    run_app = st.checkbox("Activate Recognition", value=False, key="run_app_check")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.45, key="conf_slider")
    
    st.info("💡 Scale landmarks are extracted using MediaPipe and classified using a pre-trained MLP model.")

# --- MAIN CONTENT ---
st.markdown("<h1 style='text-align: center; color: white;'>Arabic Hand Sign Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8892b0; font-size: 1.1rem; margin-bottom: 5px;'>College of Artificial Intelligence / Department of Biomedical Applications</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8892b0; font-size: 1.1rem; margin-bottom: 20px; direction: rtl;'>كلية الذكاء الاصطناعي / قسم التطبيقات الطبية الحيوية</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #00ff88; font-weight: bold;'>High-precision real-time hand sign interpretation</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

# Fixed width for standard laptop screens
CAM_WIDTH = 640 

with col1:
    st.markdown("### Camera Feed")
    FRAME_WINDOW = st.image([], width=CAM_WIDTH)

with col2:
    st.markdown("### Interpretation")
    prediction_placeholder = st.empty()
    
    def update_prediction_card(arabic_letter="---", english_letter="---", confidence=0):
        with prediction_placeholder.container():
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-title">Detected Letter</div>
                <div class="prediction-letter">{arabic_letter}</div>
                <div class="prediction-sub">{english_letter}</div>
                <div class="prediction-title" style="margin-top: 20px;">Confidence: {confidence:.1%}</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar-fill" style="width: {confidence*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    update_prediction_card()

# --- ENHANCED MODEL INFO SECTION ---
st.markdown("""
<div class="info-card">
    <div class="info-grid">
        <div class="info-text">
            <div class="info-header" style="margin-top: 0;">
                <span>🤖</span> About the Model
            </div>
            <strong>Algorithm:</strong> Multi-Layer Perceptron (MLP)<br><br>
            <strong>Why MLP?</strong> This neural network architecture was selected after rigorous benchmarking against SVM, Random Forest, KNN, and XGBoost. 
            It excels at mapping the complex 3D spatial relationships between hand landmarks, achieving an impressive 
            <span style="color: #00ff88; font-weight: bold;">96.36% accuracy</span>.
        </div>
        <div class="info-text arabic-text">
            <div class="info-header" style="margin-top: 0; justify-content: flex-end;">
                حول النموذج <span style="margin-left: 15px;">🤖</span>
            </div>
            <strong>الخوارزمية:</strong> الشبكة العصبية (MLP)<br><br>
            <strong>لماذا MLP؟</strong> تم اختيار بنية الشبكة العصبية هذه بعد إجراء مقارنات دقيقة مع خوارزميات أخرى. 
            تتفوق في فهم العلاقات المكانية المعقدة ثلاثية الأبعاد لنقاط اليد، مما أدى إلى تحقيق دقة مذهلة تصل إلى 
            <span style="color: #00ff88; font-weight: bold;">96.36٪</span>.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- LOGIC WITH RETRY ---
if run_app:
    # Give the OS a moment to release the camera if this is a rerun
    time.sleep(0.5) 
    
    cap = cv2.VideoCapture(0)
    
    # Retry logic
    retry_count = 0
    while not cap.isOpened() and retry_count < 3:
        time.sleep(1)
        cap = cv2.VideoCapture(0)
        retry_count += 1

    if not cap.isOpened():
        st.error("Camera is busy or not found. Please wait a moment and try toggling 'Activate Recognition' again.")
        if st.button("Manually Reset Connection"):
            st.rerun()
    else:
        try:
            # Main Loop
            while st.session_state.run_app_check:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to MediaPipe Image object
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
                
                # Perform hand landmark detection
                result = landmarker.detect(mp_image)
                
                english_letter = "---"
                arabic_letter = "---"
                confidence = 0
                
                if result.hand_landmarks:
                    for hand_landmarks in result.hand_landmarks:
                        # Use my custom drawing function
                        draw_landmarks(frame, hand_landmarks)
                        
                        features = extract_landmarks(hand_landmarks)
                        features_scaled = scaler.transform(features)
                        
                        probs = model.predict_proba(features_scaled)
                        pred_idx = np.argmax(probs)
                        confidence = probs[0][pred_idx]
                        
                        # Use threshold from slider
                        if confidence >= st.session_state.conf_slider:
                            english_letter = label_encoder.inverse_transform([pred_idx])[0]
                            arabic_letter = ARABIC_LABELS.get(english_letter, english_letter)
                        else:
                            english_letter = "Scanning..."
                            arabic_letter = "جاري الفحص..."

                FRAME_WINDOW.image(frame, channels="BGR", width=CAM_WIDTH)
                update_prediction_card(arabic_letter, english_letter, confidence)
                
        finally:
            cap.release()
            # No destroyAllWindows needed for Streamlit container
else:
    st.info("App paused. Enable 'Activate Recognition' in the sidebar to start the camera feed.")
