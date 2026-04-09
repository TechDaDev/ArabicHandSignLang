API_DESCRIPTION = (
    "FastAPI backend for the Arabic Hand Sign Language mobile application. "
    "This phase adds authenticated single-frame inference using the existing trained artifacts."
)

API_TAGS = [
    {"name": "health", "description": "Service and database health endpoints."},
    {"name": "auth", "description": "Registration, login, and current-user authentication endpoints."},
    {"name": "users", "description": "Authenticated user profile endpoints."},
    {"name": "predict", "description": "One-frame landmark inference endpoints."},
    {"name": "history", "description": "Authenticated prediction history endpoints."},
]

ARABIC_LABELS = {
    "Ain": "ع",
    "Al": "ال",
    "Alef": "أ",
    "Beh": "ب",
    "Dad": "ض",
    "Dal": "د",
    "Feh": "ف",
    "Ghain": "غ",
    "Hah": "ح",
    "Heh": "هـ",
    "Jeem": "ج",
    "Kaf": "ك",
    "Khah": "خ",
    "Laa": "لا",
    "Lam": "ل",
    "Meem": "م",
    "Noon": "ن",
    "Qaf": "ق",
    "Reh": "ر",
    "Sad": "ص",
    "Seen": "س",
    "Sheen": "ش",
    "Tah": "ط",
    "Teh": "ت",
    "Teh_Marbuta": "ة",
    "Theh": "ث",
    "Waw": "و",
    "Yeh": "ي",
    "Zah": "ظ",
    "Zain": "ز",
    "thal": "ذ",
}

FEATURE_NAMES = [f"{axis}{index}" for index in range(21) for axis in ("x", "y", "z")]
EXPECTED_LANDMARKS = 21
EXPECTED_FEATURES = 63
