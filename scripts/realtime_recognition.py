import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import numpy as np
import joblib
import os

# 1. Load the trained components
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

model_path = os.path.join(project_root, "models", "hand_sign_model.pkl")
scaler_path = os.path.join(project_root, "models", "scaler.pkl")
label_encoder_path = os.path.join(project_root, "models", "label_encoder.pkl")
landmarker_path = os.path.join(project_root, "hand_landmarker.task")

if not all(os.path.exists(f) for f in [model_path, scaler_path, label_encoder_path, landmarker_path]):
    print(f"Error: Required artifacts not found.")
    print(f"Looked in: {project_root}")
    print("Please ensure hand_landmarker.task is in the root and models are trained.")
    exit(1)

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

# 2. Initialize MediaPipe Task Hand Landmarker
base_options = mp.tasks.BaseOptions(model_asset_path=landmarker_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = vision.HandLandmarker.create_from_options(options)

# 3. Helper functions from streamlit_app.py
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def draw_landmarks(image, hand_landmarks_list):
    h, w, _ = image.shape
    for start_idx, end_idx in HAND_CONNECTIONS:
        start_pt = hand_landmarks_list[start_idx]
        end_pt = hand_landmarks_list[end_idx]
        cv2.line(image, 
                 (int(start_pt.x * w), int(start_pt.y * h)), 
                 (int(end_pt.x * w), int(end_pt.y * h)), 
                 (0, 255, 136), 2)
    for lm in hand_landmarks_list:
        cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 5, (255, 255, 255), -1)

def extract_landmarks(hand_landmarks_list):
    landmarks = []
    for lm in hand_landmarks_list:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks).reshape(1, -1)

# 4. Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

print("Starting real-time recognition... Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image object
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
    
    # Process the image and find hands
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Draw landmarks
            draw_landmarks(frame, hand_landmarks)

            # Extract features and scale them
            features = extract_landmarks(hand_landmarks)
            features_scaled = scaler.transform(features)

            # Predict the letter
            # Note: streamlit_app uses predict_proba, but here we keep script simple
            pred = model.predict(features_scaled)[0]
            letter = label_encoder.inverse_transform([pred])[0]

            # Display the prediction
            cv2.putText(
                frame,
                f"Letter: {letter}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )

    # Show the frame
    cv2.imshow("Arabic Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Cleanup complete.")
