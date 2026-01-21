import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# 1. Load the trained components
model_path = "hand_sign_model.pkl"
scaler_path = "scaler.pkl"
label_encoder_path = "label_encoder.pkl"

if not all(os.path.exists(f) for f in [model_path, scaler_path, label_encoder_path]):
    print("Error: Model artifacts not found. Please run train_model.py first.")
    exit(1)

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

# 2. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 3. Helper function to extract landmarks
def extract_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks).reshape(1, -1)  # shape (1, 63)

# 4. Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam. Try changing the index (e.g., VideoCapture(1)).")
    exit(1)

print("Starting real-time recognition... Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Extract features and scale them
            features = extract_landmarks(hand_landmarks)
            features_scaled = scaler.transform(features)

            # Predict the letter
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

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Cleanup complete.")
