"""HostWise shared preprocessing pipelines.

V1: eight initial features (morning baseline, unchanged).
V2: sixteen features (V1 + review, host, neighbourhood, size and stay features).
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# V1
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["accommodates", "bedrooms", "bathrooms", "latitude", "longitude"]
CATEGORICAL_FEATURES = ["city", "room_type", "property_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, NUMERIC_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
])

# ---------------------------------------------------------------------------
# V2
# ---------------------------------------------------------------------------
NUMERIC_V2 = ["accommodates", "bathrooms", "latitude", "longitude", "beds", "minimum_nights", "hosts_time_as_host_years"]
NUMERIC_FLAG_V2 = ["bedrooms", "review_scores_rating"]
LOG_NUMERIC_V2 = ["number_of_reviews", "calculated_host_listings_count"]
CATEGORICAL_V2 = ["city", "room_type", "property_type", "neighbourhood_cleansed", "host_is_superhost"]
FEATURES_V2 = NUMERIC_V2 + NUMERIC_FLAG_V2 + LOG_NUMERIC_V2 + CATEGORICAL_V2

numeric_flag_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
])

log_numeric_pipeline = Pipeline([
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_grouped_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist", sparse_output=False)),
])

preprocessor_v2 = ColumnTransformer([
    ("num", numeric_pipeline, NUMERIC_V2),
    ("num_flag", numeric_flag_pipeline, NUMERIC_FLAG_V2),
    ("num_log", log_numeric_pipeline, LOG_NUMERIC_V2),
    ("cat", categorical_grouped_pipeline, CATEGORICAL_V2),
])


# ---------------------------------------------------------------------------
# V2.1: V2 with `bathrooms` replaced by features parsed from `bathrooms_text`
# ---------------------------------------------------------------------------
def parse_bathrooms(X):
    """Stateless parse: bath count (half-bath = 0.5) and privacy (shared / private / not specified)."""
    text = X.iloc[:, 0].astype("string").str.lower()
    count = text.str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
    count = count.mask(text.str.contains("half-bath", na=False), 0.5)
    return pd.DataFrame({
        "bath_count": count,
        "bath_shared": text.str.contains("shared", na=False).astype(int),
        "bath_private": text.str.contains("private", na=False).astype(int),
    }, index=X.index)


def bathroom_feature_names(transformer, input_features):
    return ["bath_count", "bath_shared", "bath_private"]


NUMERIC_V21 = [f for f in NUMERIC_V2 if f != "bathrooms"]
BATHROOM_V21 = ["bathrooms_text"]
FEATURES_V21 = NUMERIC_V21 + NUMERIC_FLAG_V2 + LOG_NUMERIC_V2 + BATHROOM_V21 + CATEGORICAL_V2

bathroom_pipeline = Pipeline([
    ("parse", FunctionTransformer(parse_bathrooms, feature_names_out=bathroom_feature_names)),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor_v21 = ColumnTransformer([
    ("num", numeric_pipeline, NUMERIC_V21),
    ("num_flag", numeric_flag_pipeline, NUMERIC_FLAG_V2),
    ("num_log", log_numeric_pipeline, LOG_NUMERIC_V2),
    ("bath", bathroom_pipeline, BATHROOM_V21),
    ("cat", categorical_grouped_pipeline, CATEGORICAL_V2),
])


# ---------------------------------------------------------------------------
# V2.2: V2.1 + distance to city centre + 10 amenity flags (Day 3, Track A)
# ---------------------------------------------------------------------------
CITY_LAT = {"london": 51.5074, "paris": 48.8566, "rome": 41.9028, "madrid": 40.4168, "barcelona": 41.3870}
CITY_LON = {"london": -0.1278, "paris": 2.3522, "rome": 12.4964, "madrid": -3.7038, "barcelona": 2.1701}

AMENITY_PATTERNS = {
    "air_conditioning": r"air conditioning",
    "elevator": r"elevator",
    "parking": r"parking",
    "pool": r"\bpool\b(?! table)",
    "washer": r"\bwasher\b",
    "dishwasher": r"dishwasher",
    "workspace": r"workspace",
    "self_check_in": r"self check-in",
    "balcony_patio": r"balcony|patio",
    "gym": r"\bgym\b",
}


def distance_to_centre(X):
    """Stateless: great-circle distance (km) to a fixed city-centre point."""
    lat1, lon1 = np.radians(X["latitude"]), np.radians(X["longitude"])
    lat2, lon2 = np.radians(X["city"].map(CITY_LAT)), np.radians(X["city"].map(CITY_LON))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return pd.DataFrame({"dist_centre_km": 2 * 6371 * np.arcsin(np.sqrt(a))}, index=X.index)


def amenity_flags(X):
    """Stateless: 1/0 flags for a fixed list of amenities."""
    text = X["amenities"].fillna("").str.lower()
    return pd.DataFrame({name: text.str.contains(p).astype(int) for name, p in AMENITY_PATTERNS.items()},
                        index=X.index)


def distance_feature_names(transformer, input_features):
    return ["dist_centre_km"]


def amenity_feature_names(transformer, input_features):
    return list(AMENITY_PATTERNS)


FEATURES_V22 = FEATURES_V21 + ["amenities"]

distance_pipeline = Pipeline([
    ("make", FunctionTransformer(distance_to_centre, feature_names_out=distance_feature_names)),
    ("impute", SimpleImputer(strategy="median")),
])

amenity_pipeline = Pipeline([
    ("make", FunctionTransformer(amenity_flags, feature_names_out=amenity_feature_names)),
    ("impute", SimpleImputer(strategy="median")),
])

preprocessor_v22 = ColumnTransformer(preprocessor_v21.transformers + [
    ("distance", distance_pipeline, ["city", "latitude", "longitude"]),
    ("amenity_flags", amenity_pipeline, ["amenities"]),
])
