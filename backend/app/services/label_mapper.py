from app.core.constants import ARABIC_LABELS


def get_arabic_label(english_label: str) -> str:
    return ARABIC_LABELS.get(english_label, english_label)
