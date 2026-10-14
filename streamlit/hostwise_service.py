"""HostWise inference service: loads the saved pipeline and turns form inputs into a priced result.

Kept free of Streamlit so it can be tested on its own. Uses the saved regression pipeline and the
calibrated pricing classifier unchanged; no thresholds are defined here.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = PROJECT_ROOT / "shared_foundation"
MODEL_PATH = PROJECT_ROOT / "models" / "hostwise_pipeline.joblib"
META_PATH = PROJECT_ROOT / "models" / "hostwise_pipeline_meta.json"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hostwise_5cities.csv"

# `preprocessing` must be importable before the pipeline is unpickled.
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from pricing_classifier import classify_price, load_thresholds  # noqa: E402

CITIES = {"london": "London", "paris": "Paris", "rome": "Rome", "madrid": "Madrid", "barcelona": "Barcelona"}
ROOM_TYPES = ["Entire home/apt", "Private room", "Hotel room", "Shared room"]
REQUIRED = ["city", "neighbourhood_cleansed", "latitude", "longitude", "room_type", "property_type",
            "accommodates", "amenities"]
TEXT_COLUMNS = ["city", "room_type", "property_type", "neighbourhood_cleansed", "host_is_superhost",
                "bathrooms_text", "amenities"]

# The ten amenities the model uses (preprocessing.AMENITY_PATTERNS), with Inside Airbnb wording.
MODEL_AMENITIES = {
    "Air conditioning": "Air conditioning",
    "Elevator": "Elevator",
    "Parking": "Free parking on premises",
    "Pool": "Pool",
    "Washer": "Washer",
    "Dishwasher": "Dishwasher",
    "Dedicated workspace": "Dedicated workspace",
    "Self check-in": "Self check-in",
    "Balcony or patio": "Patio or balcony",
    "Gym": "Gym",
}

STATUS_LABELS = {
    "Typical": "Within the usual range",
    "Above Typical": "Above the usual range",
    "Below Typical": "Below the usual range",
    "Uncertain": "Can't assess reliably",
}
STATUS_MESSAGES = {
    "Typical": "Your price sits where the middle half of comparable listings were priced.",
    "Above Typical": "Your price is above where most comparable listings were priced. In our data, listings "
                     "above this range tended to have fewer booked nights (an association, not a guarantee).",
    "Below Typical": "Your price is below where most comparable listings were priced.",
    "Uncertain": "The comparison is not reliable for this listing; see the note below.",
}
NO_REVIEWS_WARNING = ("No reviews yet: the estimate and range are based on listings without reviews, "
                      "whose prices vary more. Treat both as a rough guide.")
HIGH_ESTIMATE_WARNING = ("This estimate is above {limit:,} EUR, where the model has too little data. "
                         "No usual range is shown.")
EXTREME_PRICE_WARNING = ("Your asking price is far outside the range seen for similar listings. "
                         "Please check the value.")


@dataclass
class Artifacts:
    pipeline: object
    meta: dict
    thresholds: dict
    inputs: list
    cat_columns: list
    encoder: object
    version_mismatch: dict


@dataclass
class Result:
    estimate_eur: float
    has_reviews: bool
    city: str
    usual_range: dict | None
    comparison: dict | None
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def estimate_display(self) -> int:
        return round_to_5(self.estimate_eur)


def round_to_5(value: float) -> int:
    return int(5 * round(value / 5))


def load_artifacts() -> Artifacts:
    """Load the saved pipeline, its metadata and the pricing thresholds (call once and cache)."""
    import joblib

    for path in (MODEL_PATH, META_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path.relative_to(PROJECT_ROOT)}")
    pipeline = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_columns = next(cols for name, _, cols in preprocessor.transformers_ if name == "cat")
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    mismatch = {}
    for lib, trained in meta.get("versions", {}).items():
        try:
            installed = importlib.import_module(lib).__version__
        except ImportError:
            installed = "not installed"
        if installed != trained:
            mismatch[lib] = {"trained": trained, "installed": installed}
    return Artifacts(pipeline, meta, load_thresholds(), list(pipeline.feature_names_in_), list(cat_columns),
                     encoder, mismatch)


# ------------------------------------------------------------------ form helpers
def bathrooms_text(count: float | None, kind: str) -> str | None:
    """Build an Inside Airbnb style bathrooms_text, e.g. '1.5 shared baths' or 'Half-bath'."""
    if count is None:
        return None
    prefix = {"Private": "private ", "Shared": "shared ", "Not specified": ""}[kind]
    if count == 0.5:
        return "Shared half-bath" if kind == "Shared" else "Half-bath"
    number = f"{count:g}"
    return f"{number} {prefix}{'bath' if count == 1 else 'baths'}"


def amenities_text(selected: list[str]) -> str:
    return json.dumps([MODEL_AMENITIES[a] for a in selected])


def listing_options(artifacts: Artifacts) -> dict:
    """Neighbourhoods (with typical coordinates) per city, city bounds, and property types per room type."""
    property_types = [c for c in artifacts.encoder.categories_[artifacts.cat_columns.index("property_type")]
                      if c != "infrequent_sklearn"]
    infrequent = artifacts.encoder.infrequent_categories_[artifacts.cat_columns.index("property_type")]
    frequent_types = [c for c in property_types if infrequent is None or c not in set(infrequent)]
    options = {"cities": {}, "property_types": {rt: frequent_types for rt in ROOM_TYPES}, "source": "model"}
    if not DATA_PATH.exists():
        return options
    cols = ["city", "neighbourhood_cleansed", "latitude", "longitude", "room_type", "property_type"]
    data = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)
    for city in CITIES:
        part = data[data["city"] == city]
        hoods = (part.groupby("neighbourhood_cleansed")
                 .agg(listings=("latitude", "size"), lat=("latitude", "median"), lon=("longitude", "median"))
                 .sort_values("listings", ascending=False))
        options["cities"][city] = {
            "neighbourhoods": {h: {"lat": round(r.lat, 5), "lon": round(r.lon, 5), "listings": int(r.listings)}
                               for h, r in hoods.iterrows()},
            "bounds": {"lat": [float(part["latitude"].min()) - 0.05, float(part["latitude"].max()) + 0.05],
                       "lon": [float(part["longitude"].min()) - 0.05, float(part["longitude"].max()) + 0.05]},
        }
    pairs = data.groupby(["room_type", "property_type"]).size()
    for rt in ROOM_TYPES:
        if rt in pairs.index.get_level_values(0):
            ranked = pairs.loc[rt].sort_values(ascending=False)
            options["property_types"][rt] = [p for p in ranked.index if p in set(frequent_types)] or frequent_types
    options["source"] = "data"
    return options


# ------------------------------------------------------------------ validation and prediction
def validate(listing: dict, options: dict) -> list[str]:
    """Friendly messages for missing or implausible required inputs."""
    errors = []
    names = {"city": "City", "neighbourhood_cleansed": "Neighbourhood", "latitude": "Latitude",
             "longitude": "Longitude", "room_type": "Room type", "property_type": "Property type",
             "accommodates": "Number of guests", "amenities": "Amenities"}
    for key in REQUIRED:
        value = listing.get(key)
        if value is None or (isinstance(value, str) and not value.strip()) or (
                isinstance(value, float) and math.isnan(value)):
            errors.append(f"{names[key]} is required.")
    city = str(listing.get("city") or "").strip().lower()
    if city and city not in CITIES:
        errors.append(f"City must be one of: {', '.join(CITIES.values())}.")
    acc = listing.get("accommodates")
    if acc is not None and not (1 <= acc <= 16):
        errors.append("Number of guests must be between 1 and 16.")
    bounds = options.get("cities", {}).get(city, {}).get("bounds")
    lat, lon = listing.get("latitude"), listing.get("longitude")
    if bounds and lat is not None and lon is not None:
        if not (bounds["lat"][0] <= lat <= bounds["lat"][1] and bounds["lon"][0] <= lon <= bounds["lon"][1]):
            errors.append(f"The coordinates are outside {CITIES[city]}. Please check the location.")
    rating = listing.get("review_scores_rating")
    if rating is not None and not (0 <= rating <= 5):
        errors.append("Rating must be between 0 and 5.")
    return errors


def unseen_categories(artifacts: Artifacts, listing: dict) -> list[str]:
    notes = []
    for col, seen, rare in zip(artifacts.cat_columns, artifacts.encoder.categories_,
                               artifacts.encoder.infrequent_categories_):
        value = listing.get(col)
        if value is not None and value not in set(seen):
            how = "treated as a rare category" if rare is not None else "not recognised by the model"
            notes.append(f"'{value}' ({col.replace('_', ' ')}) was not seen in training and is {how}.")
    return notes


def predict_eur(artifacts: Artifacts, listing: dict) -> float:
    row = {c: np.nan if listing.get(c) is None else listing[c] for c in artifacts.inputs}
    X = pd.DataFrame([row])
    present = [c for c in TEXT_COLUMNS if c in X.columns]
    X[present] = X[present].astype(object)
    return max(float(artifacts.pipeline.predict(X)[0]), 0.0)


def usual_range(artifacts: Artifacts, estimate: float, has_reviews: bool) -> dict | None:
    """Calibrated range from pricing_thresholds.json (same values classify_price uses)."""
    if estimate > artifacts.thresholds["max_reliable_estimate_eur"]:
        return None
    seg = artifacts.thresholds["segments"]["with_reviews" if has_reviews else "no_reviews"]
    return {"lower": round(estimate * seg["typical_low"], 2), "upper": round(estimate * seg["typical_high"], 2)}


def assess(artifacts: Artifacts, listing: dict, asking_price_eur: float | None = None) -> Result:
    """Estimate the nightly price and, if an asking price is given, compare it."""
    listing = {**listing, "city": str(listing.get("city", "")).strip().lower()}
    has_reviews = bool(listing.get("number_of_reviews"))
    if not has_reviews:                       # new listing: no reviews, rating left empty, as in training
        listing.update(number_of_reviews=0, review_scores_rating=None)

    estimate = predict_eur(artifacts, listing)
    rng = usual_range(artifacts, estimate, has_reviews)
    comparison = classify_price(estimate, asking_price_eur, has_reviews) if asking_price_eur is not None else None

    warnings = []
    if not has_reviews:
        warnings.append(NO_REVIEWS_WARNING)
    if rng is None:
        warnings.append(HIGH_ESTIMATE_WARNING.format(limit=artifacts.thresholds["max_reliable_estimate_eur"]))
    if comparison and comparison["warning"] and "far outside" in comparison["warning"]:
        warnings.append(EXTREME_PRICE_WARNING)
    return Result(estimate, has_reviews, listing["city"], rng, comparison, warnings,
                  unseen_categories(artifacts, listing))
