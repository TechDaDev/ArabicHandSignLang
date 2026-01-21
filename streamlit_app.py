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
        font-family: 'Orbitron', sans-serif;
        font-size: 5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
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
    model = joblib.load("hand_sign_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_assets()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# --- MEDIAPIPE SETUP (CACHED) ---
@st.cache_resource
def get_hands_model():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

hands = get_hands_model()
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def extract_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks).reshape(1, -1)

# --- SIDEBAR ---
with st.sidebar:
    logo_path = "/home/zeus3000/PycharmProjects/hand_signs/logo/coai_logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=300)
    
    st.title("Settings")
    # Using key to preserve state across reruns better
    run_app = st.checkbox("Activate Recognition", value=False, key="run_app_check")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.45, key="conf_slider")
    
    st.info("💡 Scale landmarks are extracted using MediaPipe and classified using a pre-trained MLP model.")
    
    st.markdown("---")
    st.markdown("### 🤖 About the Model / حول النموذج")
    
    st.markdown("""
    **Algorithm:** Multi-Layer Perceptron (MLP)
    
    **Why?** MLP was chosen after benchmarking against SVM, Random Forest, KNN, and XGBoost. It effectively captures complex patterns in landmark coordinates, delivering the highest accuracy (96.36%).
    """)
    
    st.markdown("""
    <div style="direction: rtl; text-align: right;">
    <b>الخوارزمية:</b> الشبكة العصبية (MLP)<br>
    <b>لماذا؟</b> تم اختيار MLP بعد المقارنة مع عدة نماذج. تتميز بقدرتها العالية على فهم الأنماط المعقدة لنقاط اليد، وحققت أعلى دقة (96.36٪).
    </div>
    """, unsafe_allow_html=True)

# --- MAIN CONTENT ---
st.markdown("<h1 style='text-align: center; color: white;'>Arabic Sign Language Recognition</h1>", unsafe_allow_html=True)
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
    
    def update_prediction_card(letter="---", confidence=0):
        with prediction_placeholder.container():
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-title">Detected Letter</div>
                <div class="prediction-letter">{letter}</div>
                <div class="prediction-title" style="margin-top: 20px;">Confidence: {confidence:.1%}</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar-fill" style="width: {confidence*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    update_prediction_card()

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
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                
                current_letter = "---"
                confidence = 0
                
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        features = extract_landmarks(hand_landmarks)
                        features_scaled = scaler.transform(features)
                        
                        probs = model.predict_proba(features_scaled)
                        pred_idx = np.argmax(probs)
                        confidence = probs[0][pred_idx]
                        
                        # Use threshold from slider
                        if confidence >= st.session_state.conf_slider:
                            current_letter = label_encoder.inverse_transform([pred_idx])[0]
                        else:
                            current_letter = "Scanning..."

                FRAME_WINDOW.image(frame, channels="BGR", width=CAM_WIDTH)
                update_prediction_card(current_letter, confidence)
                
        finally:
            cap.release()
            # No destroyAllWindows needed for Streamlit container
else:
    st.info("App paused. Enable 'Activate Recognition' in the sidebar to start the camera feed.")
