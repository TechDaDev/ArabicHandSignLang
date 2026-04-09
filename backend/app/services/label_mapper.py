from app.core.constants import ARABIC_LABELS


def get_arabic_label(label: str) -> str:
    """Return the Arabic equivalent for a model label when available."""
    return ARABIC_LABELS.get(label, label)
