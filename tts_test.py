from gtts import gTTS
from playsound import playsound
import tempfile
import os


def test_tts():
    text = "مرحبا بك في مشروع لغة الإشارة العربية"
    print("Testing online Arabic TTS with gTTS...")
    print(f"Speaking: {text}")

    tmp_name = None
    try:
        tts = gTTS(text=text, lang="ar")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tmp_name = f.name
        tts.save(tmp_name)
        playsound(tmp_name)
        print("✅ SUCCESS: Online Arabic TTS played successfully.")
    except Exception as e:
        print("❌ ERROR: Online TTS failed.")
        print("Details:", e)
        print("Check your internet connection and package installation.")
    finally:
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass

if __name__ == "__main__":
    test_tts()
