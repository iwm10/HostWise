"""Build streamlit/app_options.json from the full training dataset (run once, offline).

The app reads only this small file, so the 600 MB dataset is not needed at runtime or in the repository.
Run from the project root, with data/processed/hostwise_5cities.csv present:
    python streamlit/build_app_options.py
"""
import json

import pandas as pd

import hostwise_service as svc

DATA_PATH = svc.PROJECT_ROOT / "data" / "processed" / "hostwise_5cities.csv"


def build_options(artifacts: svc.Artifacts, data: pd.DataFrame) -> dict:
    frequent_types = svc.model_property_types(artifacts)
    options = {"cities": {}, "property_types": {rt: frequent_types for rt in svc.ROOM_TYPES}, "source": "data"}
    for city in svc.CITIES:
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
    for rt in svc.ROOM_TYPES:
        if rt in pairs.index.get_level_values(0):
            ranked = pairs.loc[rt].sort_values(ascending=False)
            options["property_types"][rt] = [p for p in ranked.index if p in set(frequent_types)] or frequent_types
    return options


if __name__ == "__main__":
    cols = ["city", "neighbourhood_cleansed", "latitude", "longitude", "room_type", "property_type"]
    data = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)
    options = build_options(svc.load_artifacts(), data)
    svc.OPTIONS_PATH.write_text(json.dumps(options, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {svc.OPTIONS_PATH.relative_to(svc.PROJECT_ROOT)}: "
          + ", ".join(f"{c} {len(v['neighbourhoods'])}" for c, v in options["cities"].items()) + " neighbourhoods")
