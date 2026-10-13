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
