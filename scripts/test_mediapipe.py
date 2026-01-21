import mediapipe as mp

try:
    print(f"MediaPipe Version: {mp.__version__}")
    print(f"Solutions module: {mp.solutions}")
    print(f"Hands module: {mp.solutions.hands}")
    print("SUCCESS: mediapipe is correctly installed.")
except Exception as e:
    print(f"ERROR: {e}")
