from __future__ import annotations

import logging
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.services.label_mapper import get_arabic_label


logger = logging.getLogger(__name__)


def _parse_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _get_candidate_label(record: Any) -> tuple[str | None, str | None]:
    top_predictions = getattr(record, "top_predictions_json", None) or []
    if top_predictions:
        top_entry = top_predictions[0]
        label = top_entry.get("label")
        if label:
            return label, top_entry.get("arabic_label") or get_arabic_label(label)

    label = getattr(record, "predicted_label", None)
    if not label or label == "Scanning...":
        return None, None
    return label, getattr(record, "arabic_label", None) or get_arabic_label(label)


def _get_stable_label(history: list[str]) -> tuple[str | None, int]:
    if len(history) < int(settings.SESSION_STABLE_MIN_HISTORY):
        return None, 0

    counts = Counter(history)
    label, count = counts.most_common(1)[0]
    return str(label), int(count)


def _should_commit_label(
    stable_label: str | None,
    stable_count: int,
    now: datetime,
    last_commit_time: datetime | None,
    last_committed_label: str | None,
) -> bool:
    if stable_label is None:
        return False
    if stable_count < int(settings.SESSION_STABLE_MIN_COUNT):
        return False
    if last_commit_time is not None and (now - last_commit_time).total_seconds() < float(settings.SESSION_COMMIT_COOLDOWN_SECONDS):
        return False
    if stable_label == last_committed_label:
        return False
    return True


def build_session_recognition_state(prediction_records: list[Any]) -> dict[str, Any]:
    """Reconstruct script-like stabilized recognition state from persisted session predictions."""
    recent_raw_predictions: deque[str] = deque(maxlen=int(settings.SESSION_PREDICTION_WINDOW_SIZE))

    current_stable_label: str | None = None
    current_stable_arabic_label: str | None = None
    current_stable_count = 0
    is_stable = False

    last_stable_label: str | None = None
    last_stable_arabic_label: str | None = None
    last_committed_label: str | None = None
    last_committed_arabic_label: str | None = None
    last_commit_time: datetime | None = None

    current_word = ""
    text_buffer = ""

    ordered_records = sorted(
        prediction_records,
        key=lambda item: _parse_datetime(getattr(item, "created_at", None)) or datetime.min.replace(tzinfo=timezone.utc),
    )

    for record in ordered_records:
        record_time = _parse_datetime(getattr(record, "created_at", None)) or datetime.now(timezone.utc)

        if current_word and last_commit_time is not None:
            idle_seconds = (record_time - last_commit_time).total_seconds()
            if idle_seconds >= float(settings.SESSION_AUTO_SPACE_TIMEOUT_SECONDS):
                if text_buffer and not text_buffer.endswith(" "):
                    text_buffer += " "
                current_word = ""
                last_committed_label = None
                last_committed_arabic_label = None
                recent_raw_predictions.clear()

        candidate_label, candidate_arabic_label = _get_candidate_label(record)
        if candidate_label is None or candidate_arabic_label is None:
            continue

        recent_raw_predictions.append(candidate_label)
        stable_label, stable_count = _get_stable_label(list(recent_raw_predictions))

        current_stable_count = stable_count
        is_stable = stable_label is not None and stable_count >= int(settings.SESSION_STABLE_MIN_COUNT)

        if is_stable and stable_label is not None:
            current_stable_label = stable_label
            current_stable_arabic_label = get_arabic_label(stable_label)
            last_stable_label = current_stable_label
            last_stable_arabic_label = current_stable_arabic_label

            if _should_commit_label(
                stable_label=current_stable_label,
                stable_count=current_stable_count,
                now=record_time,
                last_commit_time=last_commit_time,
                last_committed_label=last_committed_label,
            ):
                text_buffer += current_stable_arabic_label
                current_word += current_stable_arabic_label
                last_committed_label = current_stable_label
                last_committed_arabic_label = current_stable_arabic_label
                last_commit_time = record_time

                logger.info(
                    "Session stabilization committed label=%s stable_count=%s text_buffer=%s",
                    current_stable_label,
                    current_stable_count,
                    text_buffer,
                )
        else:
            current_stable_label = None
            current_stable_arabic_label = None

    return {
        "recent_raw_predictions_window": list(recent_raw_predictions),
        "stable_label": current_stable_label,
        "stable_arabic_label": current_stable_arabic_label,
        "is_stable": is_stable,
        "stable_count": int(current_stable_count),
        "current_word": current_word,
        "text_buffer": text_buffer,
        "last_stable_label": last_stable_label,
        "last_stable_arabic_label": last_stable_arabic_label,
        "last_committed_label": last_committed_label,
        "last_committed_arabic_label": last_committed_arabic_label,
        "last_commit_timestamp": last_commit_time,
    }
