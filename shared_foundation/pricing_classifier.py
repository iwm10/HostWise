"""HostWise pricing classification.

Compares a host's listed nightly price (EUR) with the regression model's estimate.
Thresholds are calibrated in 06_pricing_classification.ipynb from out-of-fold training
predictions and stored in models/pricing_thresholds.json.

Labels describe where the listed price sits relative to typical prices of similar training
listings. They are not fair market values or confidence intervals.
"""
import json
import math
import numbers
from functools import lru_cache
from pathlib import Path

DEFAULT_THRESHOLDS = Path(__file__).resolve().parents[1] / "models" / "pricing_thresholds.json"

BELOW, TYPICAL, ABOVE, UNCERTAIN = "Below Typical", "Typical", "Above Typical", "Uncertain"
GROUP_TEXT = {"with_reviews": "with reviews", "no_reviews": "without reviews"}


@lru_cache(maxsize=None)
def load_thresholds(path=DEFAULT_THRESHOLDS):
    """Read the calibrated thresholds once and cache them."""
    return json.loads(Path(path).read_text())


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return float(value)


def classify_price(predicted_price_eur, listed_price_eur, has_reviews, thresholds=None):
    """Classify a listed nightly price against the model estimate.

    Args:
        predicted_price_eur: model estimate for the listing (EUR per night).
        listed_price_eur: the host's proposed price (EUR per night).
        has_reviews: True if the listing has at least one review.
        thresholds: optional dict; defaults to models/pricing_thresholds.json.

    Returns:
        JSON-serializable dict. Raises ValueError for invalid prices.
    """
    predicted = _positive_number(predicted_price_eur, "predicted_price_eur")
    listed = _positive_number(listed_price_eur, "listed_price_eur")
    config = thresholds or load_thresholds()
    group = "with_reviews" if has_reviews else "no_reviews"
    band = config["segments"][group]

    ratio = listed / predicted
    typical_range = {"lower": round(predicted * band["typical_low"], 2),
                     "upper": round(predicted * band["typical_high"], 2)}
    warnings = []

    if predicted > config["max_reliable_estimate_eur"]:
        status, typical_range = UNCERTAIN, None
        warnings.append(f"The estimate is above {config['max_reliable_estimate_eur']} EUR, "
                        "where the model has too little data for a reliable comparison.")
    elif not band["extreme_low"] <= ratio <= band["extreme_high"]:
        status = UNCERTAIN
        warnings.append("The listed price is far outside the range seen for similar listings; please check the value.")
    elif ratio < band["typical_low"]:
        status = BELOW
    elif ratio > band["typical_high"]:
        status = ABOVE
    else:
        status = TYPICAL

    if not has_reviews:
        warnings.append("No reviews yet: the estimate and range are based on listings without reviews, "
                        "whose prices vary more.")

    return {
        "predicted_price_eur": round(predicted, 2),
        "listed_price_eur": round(listed, 2),
        "typical_range_eur": typical_range,
        "pricing_status": status,
        "difference_eur": round(listed - predicted, 2),
        "difference_percent": round(100 * (listed - predicted) / predicted, 2),
        "has_reviews": bool(has_reviews),
        "basis": f"Typical = middle 50% of training listings {GROUP_TEXT[group]}, relative to their model estimate",
        "warning": " ".join(warnings) or None,
    }
