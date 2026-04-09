from app.models.prediction_record import PredictionRecord
from app.models.saved_phrase import SavedPhrase


def build_session_summary(
    records: list[PredictionRecord],
    saved_phrases: list[SavedPhrase],
) -> dict[str, float | int | str | None]:
    total_predictions = len(records)
    average_confidence = None
    if total_predictions:
        average_confidence = round(sum(record.confidence for record in records) / total_predictions, 4)

    latest_phrase = saved_phrases[-1].phrase if saved_phrases else None

    return {
        "total_predictions": total_predictions,
        "average_confidence": average_confidence,
        "latest_phrase": latest_phrase,
    }
