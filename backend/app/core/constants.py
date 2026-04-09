API_TAGS = [
    {"name": "auth", "description": "Authentication and token management."},
    {"name": "users", "description": "Current user profile endpoints."},
    {"name": "predict", "description": "Arabic hand sign inference endpoints."},
    {"name": "sessions", "description": "Prediction session management."},
    {"name": "history", "description": "Prediction history and saved phrases."},
    {"name": "feedback", "description": "User feedback collection."},
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

FEATURE_NAMES = [axis + str(i) for i in range(21) for axis in ("x", "y", "z")]
LANDMARK_COUNT = 21
LANDMARK_VECTOR_SIZE = 63
