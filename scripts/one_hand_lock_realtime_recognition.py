import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import numpy as np
import joblib
import os
import tempfile
import pandas as pd
import time
import threading
from gtts import gTTS
from playsound import playsound
from collections import deque, Counter
from PIL import Image as PilImage, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

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

# 2. Arabic Label Mapping from streamlit_app.py
ARABIC_LABELS = {
    "Ain": "ع", "Al": "ال", "Alef": "ا", "Beh": "ب", "Dad": "ض",
    "Dal": "د", "Feh": "ف", "Ghain": "غ", "Hah": "ح", "Heh": "هـ",
    "Jeem": "ج", "Kaf": "ك", "Khah": "خ", "Laa": "لا", "Lam": "ل",
    "Meem": "م", "Noon": "ن", "Qaf": "ق", "Reh": "ر", "Sad": "ص",
    "Seen": "س", "Sheen": "ش", "Tah": "ط", "Teh": "ت", "Teh_Marbuta": "ة",
    "Theh": "ث", "Waw": "و", "Yeh": "ي", "Zah": "ظ", "Zain": "ز", "thal": "ذ"
}

# 3. Initialize MediaPipe Task Hand Landmarker in VIDEO mode
base_options = mp.tasks.BaseOptions(model_asset_path=landmarker_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,  # allow multiple hands
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = vision.HandLandmarker.create_from_options(options)

# 3. Helper functions
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
    
    # Create feature names to match training (x0, y0, z0, ..., x20, y20, z20)
    feature_names = []
    for i in range(21):
        feature_names.extend([f'x{i}', f'y{i}', f'z{i}'])
        
    return pd.DataFrame([landmarks], columns=feature_names)

def hand_centroid(hand_landmarks):
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    return (float(np.mean(xs)), float(np.mean(ys)))

def hand_bbox_area(hand_landmarks):
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return float(w * h)

def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def select_hand_to_lock(hands, frame_center=(0.5, 0.5)):
    # Score: prefer large + near center
    best_i, best_score = 0, -1e9
    for i, hl in enumerate(hands):
        c = hand_centroid(hl)
        area = hand_bbox_area(hl)
        center_penalty = dist2(c, frame_center)  # smaller is better
        score = (area * 2.0) - (center_penalty * 0.8)
        if score > best_score:
            best_score = score
            best_i = i
    return best_i

# 4. Arabic Text Rendering Helper
def draw_arabic_text(img, text, position, font_size=40, color=(0, 255, 0)):
    # 1. Reshape and BiDi only for Arabic text
    if any('\u0600' <= c <= '\u06FF' for c in text):
        reshaped_text = arabic_reshaper.reshape(text)
        display_text = get_display(reshaped_text)
    else:
        display_text = text
    
    # 2. Convert Opencv image to PIL
    img_pil = PilImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 3. Load font (Windows default)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
        
    # 4. Draw text
    draw.text(position, display_text, font=font, fill=color)
    
    # 5. Convert back to Opencv
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 5. Stabilization Helper functions
def get_stable_letter(history):
    """
    Returns (stable_letter, count) or (None, 0)
    """
    if len(history) < 5:
        return None, 0
    counts = Counter(history)
    letter, cnt = counts.most_common(1)[0]
    return letter, cnt

def should_commit_letter(stable_letter, stable_count, now, last_commit_time, last_committed_letter):
    if stable_letter is None:
        return False
    if stable_count < STABLE_MIN_COUNT:
        return False
    if (now - last_commit_time) < COMMIT_COOLDOWN:
        return False
    if stable_letter == last_committed_letter:
        return False
    return True

def norm_to_px(x_norm, y_norm, frame_w, frame_h):
    return int(x_norm * frame_w), int(y_norm * frame_h)

def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return (x1 <= px <= x2) and (y1 <= py <= y2)

def draw_clear_box(frame, rect, progress=0.0):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(frame, "CLEAR", (x1 + 10, y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # progress bar under the box
    bar_h = 8
    bar_w = int((x2 - x1) * max(0.0, min(1.0, progress)))
    cv2.rectangle(frame, (x1, y2 - bar_h), (x1 + bar_w, y2), (255, 255, 255), -1)

# 5. TTS Helper function
def speak_ar(text):
    text = text.strip()
    if not text:
        return

    def _run():
        try:
            tts = gTTS(text=text, lang="ar")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tmp_name = f.name
            tts.save(tmp_name)
            playsound(tmp_name)
            os.remove(tmp_name)
        except Exception as e:
            print("TTS error:", e)

    threading.Thread(target=_run, daemon=True).start()

# 5. State variables for hand locking
locked = False
locked_centroid = None
lost_frames = 0
MAX_LOST = 8              # how many frames allowed without good match
RELOCK_DIST2 = 0.02       # normalized distance threshold (tune)

# ===== Letter stabilization + word building state =====
PRED_WINDOW = 12              # frames for majority vote (10–15 is good)
STABLE_MIN_COUNT = 8          # must appear >= this count in the window to be considered stable
COMMIT_COOLDOWN = 0.75        # seconds (prevents repeated commits when holding the same sign)
AUTO_SPACE_AFTER = 3.0        # seconds without commits -> add space (end word)

pred_history = deque(maxlen=PRED_WINDOW)

last_committed_letter = None
last_commit_time = 0.0

text_buffer = ""              # final output stream (words + spaces)
current_word = ""             # optional: track current word separately if you want

# ===== Clear box (touch & hold) =====
CLEAR_HOLD_SECONDS = 1.0

clear_hold_start = None     # time when fingertip entered box
clear_triggered = False     # prevent repeated clearing while holding

# ===== TTS State & Config =====
SPEAK_LETTERS = False
SPEAK_WORDS = True
SPEAK_COOLDOWN = 0.8       # avoid overlapping sounds too much
last_speak_time = 0.0

# ===== Speak-on-hand-missing timer =====
NO_HAND_SPEAK_AFTER = 1.0   # seconds with no hand -> speak current word
no_hand_start = None
word_spoken_for_current = False

# 5. Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

print("Starting real-time recognition with Hand Lock... Press 'ESC' to exit.")

# Use millisecond wall-clock for MediaPipe Video mode
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Top-right clear box (tune sizes if needed)
    box_w, box_h = 180, 80
    margin = 15
    CLEAR_RECT = (w - box_w - margin, margin, w - margin, margin + box_h)

    # draw clear box (progress updated later)
    draw_clear_box(frame, CLEAR_RECT, progress=0.0)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image object
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
    
    # Process the image with frame timestamp for VIDEO mode
    # Must be monotonically increasing
    timestamp_ms = int((time.time() - start_time) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    stable_letter = None

    if result.hand_landmarks:
        no_hand_start = None
        hands_found = result.hand_landmarks

        # 1) Acquire lock if not locked
        if not locked:
            idx = select_hand_to_lock(hands_found, frame_center=(0.5, 0.5))
            locked = True
            locked_centroid = hand_centroid(hands_found[idx])
            lost_frames = 0
        else:
            # 2) Maintain lock: pick nearest centroid to the last locked centroid
            dists = [dist2(hand_centroid(h), locked_centroid) for h in hands_found]
            idx = int(np.argmin(dists))

            if dists[idx] > RELOCK_DIST2:
                lost_frames += 1
                if lost_frames >= MAX_LOST:
                    locked = False
                    locked_centroid = None
            else:
                lost_frames = 0
                locked_centroid = hand_centroid(hands_found[idx])

        # 3) If locked, run prediction ONLY on locked hand
        if locked and locked_centroid is not None:
            hand_landmarks = hands_found[idx]

            draw_landmarks(frame, hand_landmarks)

            features = extract_landmarks(hand_landmarks)
            features_scaled = scaler.transform(features)

            pred = model.predict(features_scaled)[0]
            english_letter = label_encoder.inverse_transform([pred])[0]
            arabic_letter = ARABIC_LABELS.get(english_letter, english_letter)

            # --- 1) push prediction to history
            pred_history.append(arabic_letter)

            # --- 2) compute stable letter
            stable_letter, stable_count = get_stable_letter(pred_history)

            now = time.time()

            # --- 3) commit stable letter
            if should_commit_letter(stable_letter, stable_count, now, last_commit_time, last_committed_letter):
                text_buffer += stable_letter
                current_word += stable_letter

                last_committed_letter = stable_letter
                last_commit_time = now
                word_spoken_for_current = False

                # --- TTS: Speak letter ---
                if SPEAK_LETTERS and (now - last_speak_time) >= SPEAK_COOLDOWN:
                    speak_ar(stable_letter)
                    last_speak_time = now

            # --- Touch & hold logic ---
            # Use index fingertip (landmark 8) as the "touch" point
            tip = hand_landmarks[8]
            tip_x, tip_y = norm_to_px(tip.x, tip.y, w, h)

            # Visualize fingertip
            cv2.circle(frame, (tip_x, tip_y), 10, (255, 255, 255), 2)

            # Touch & hold on CLEAR box
            if point_in_rect(tip_x, tip_y, CLEAR_RECT):
                if clear_hold_start is None:
                    clear_hold_start = now
                    clear_triggered = False

                hold_time = now - clear_hold_start
                progress = min(hold_time / CLEAR_HOLD_SECONDS, 1.0)

                # update the clear box with progress bar
                draw_clear_box(frame, CLEAR_RECT, progress=progress)

                if hold_time >= CLEAR_HOLD_SECONDS and not clear_triggered:
                    # ===== CLEAR ACTION =====
                    text_buffer = ""
                    current_word = ""
                    last_committed_letter = None
                    pred_history.clear()

                    clear_triggered = True  # avoid repeated clears while still inside box
            else:
                clear_hold_start = None
                clear_triggered = False
                draw_clear_box(frame, CLEAR_RECT, progress=0.0)

            # --- 4) auto-space after 3s without commits (end word)
            if locked and len(current_word) > 0 and (now - last_commit_time) >= AUTO_SPACE_AFTER:
                # --- TTS: Speak word ---
                if SPEAK_WORDS and (now - last_speak_time) >= SPEAK_COOLDOWN:
                    speak_ar(current_word)
                    last_speak_time = now
                    word_spoken_for_current = True

                text_buffer += " "
                current_word = ""
                last_committed_letter = None   # allow same letter to start next word
                pred_history.clear()           # reset stability window for clean next word

    else:
        now = time.time()

        # No hands detected → keep text, only manage lock
        if locked:
            lost_frames += 1
            if lost_frames >= MAX_LOST:
                locked = False
                locked_centroid = None

        pred_history.clear()
        clear_hold_start = None
        clear_triggered = False

        # --- Start/continue no-hand timer ---
        if no_hand_start is None:
            no_hand_start = now

        # If hand missing for 2s, speak current word (once)
        if (len(current_word) > 0
            and not word_spoken_for_current
            and (now - no_hand_start) >= NO_HAND_SPEAK_AFTER
            and (now - last_speak_time) >= SPEAK_COOLDOWN):

            if SPEAK_WORDS:
                speak_ar(current_word)
                last_speak_time = now

            word_spoken_for_current = True

            # optional: finalize the word immediately when spoken
            text_buffer += " "
            current_word = ""
            last_committed_letter = None

    # --- 5) Display English UI + Arabic output separately
    stable_disp = stable_letter if stable_letter is not None else "-"
    ui_text = f"Stable: {stable_disp}"
    cv2.putText(frame, ui_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    frame = draw_arabic_text(frame, text_buffer, (30, 80), font_size=45)

    # Show the frame
    cv2.imshow("Arabic Hand Sign Recognition (Locked)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
print("Cleanup complete.")
