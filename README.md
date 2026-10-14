# HostWise 🏠

### Airbnb Nightly Price Prediction & Comparison

HostWise is a machine learning project that estimates **nightly Airbnb listing prices** in five European cities: **London, Paris, Rome, Madrid, and Barcelona**. It also lets hosts compare their proposed asking price against the model estimate and a typical range derived from historical listing data.

> **Important:** HostWise predicts *advertised nightly prices*, not confirmed booking prices or an objectively verified “fair” market value. Predictions are estimates, not pricing guarantees.

## Project at a Glance

| Item | Details |
|---|---|
| Data source | [Inside Airbnb](https://insideairbnb.com/get-the-data/) listing snapshots (June 2026) |
| Cities | London, Paris, Rome, Madrid, Barcelona |
| Modeling dataset | 153,812 listings after scope filters |
| Train / test | 123,049 / 30,763 listings (80/20, stratified by city) |
| Raw model inputs | 17 |
| Final approach | V2.2 preprocessing + LightGBM/XGBoost equal-weight ensemble |
| Cross-validation MAE | **€87.97** (5 folds) |
| Held-out test MAE | **€108.26** |
| Held-out test median absolute error | **€33.47** |

The cross-validation MAE improved from **€118.59** for the HistGradientBoosting V2.1 baseline to **€87.97** for the final pipeline — approximately **26% lower average error** under cross-validation. The held-out test MAE is substantially affected by one implausible listing asking approximately €606,653 per night; the official test result remains €108.26.

## What HostWise Does

**1. Estimate a nightly price.** A user enters a property's city and neighborhood, location, property and room type, guest capacity, bedrooms, bathroom details, amenities, and other listing characteristics. The saved model predicts a nightly asking price.

**2. Compare an asking price (optional).** If the user supplies their own proposed nightly price, the app shows its difference from the model estimate and a calibrated **usual observed range**. The comparison can display one of four statuses:

- **Within the usual range**
- **Above the usual range**
- **Below the usual range**
- **Can't assess reliably**

The comparison status is **rule-based**, calibrated using out-of-fold predictions from the training data; it is **not a separately trained classification model**. The range is wider for listings without reviews, and uncertain cases are flagged rather than presented as reliable pricing advice.

## Machine Learning Workflow

1. **Scope and cleaning:** Keep current `city scrape` listings with a valid positive nightly price and `minimum_nights <= 29`.
2. **Currency normalization:** Convert London GBP prices to EUR using the fixed project rate **£1 = €1.15**; all other cities use EUR.
3. **Data splitting:** Use a city-stratified 80/20 train/test split with `random_state=42`; keep the held-out test set separate from model selection.
4. **Preprocessing (V2.2):** Imputation, scaling, count transforms, categorical encoding, parsing bathroom descriptions, Haversine distance to city center, and 10 amenity indicators.
5. **Model experiments:** Compare baselines and boosting models using common 5-fold cross-validation splits.
6. **Final regression:** Average LightGBM and XGBoost predictions, each trained on a `log1p(price)` target and converted back to EUR.
7. **Price comparison:** Calibrate typical price bands from out-of-fold training predictions, grouped by whether a listing has reviews.

The reported held-out test score was calculated once for final evaluation, not used to tune the model.

## Repository Structure

```text
HostWise/
├── notebooks/
│   ├── day1/                     # Data understanding and EDA
│   ├── day2/                     # Preprocessing and model comparisons
│   └── day3/                     # Final integration, classification, error analysis
│       ├── 05_integration_final.ipynb
│       └── pricing_classification.ipynb
├── shared/                       # Earlier shared project resources
├── shared_foundation/
│   ├── preprocessing.py          # Reusable V2.2 preprocessing
│   └── pricing_classifier.py     # Calibrated comparison logic
├── models/
│   ├── hostwise_pipeline.joblib
│   ├── hostwise_pipeline_meta.json
│   └── pricing_thresholds.json
├── data/
│   └── calibration/
│       └── oof_v22.csv           # Out-of-fold calibration data
├── streamlit/
│   ├── streamlit_app.py          # Demo entry point
│   ├── hostwise_service.py       # Model loading, input helpers, prediction
│   ├── check_app.py              # Application checks
│   ├── requirements.txt
│   └── README.md
├── .streamlit/
│   └── config.toml
└── README.md
```

## Streamlit Demo — Local Setup

The Streamlit code and trained model are included in this repository. Use Python in an isolated environment:

```bash
# Clone the repository
 git clone https://github.com/iwm10/HostWise.git
 cd HostWise

# Create and activate an environment (Linux/macOS)
 python -m venv .venv
 source .venv/bin/activate

# Install app dependencies
 pip install -r streamlit/requirements.txt

# Start the app
 streamlit run streamlit/streamlit_app.py
```

**Current setup limitation:** The app's `listing_options()` helper reads `data/processed/hostwise_5cities.csv` to populate neighborhoods, coordinates, and property types. That dataset is **not currently included in the GitHub repository**. Without it, the application may not provide the neighborhood selections needed for a complete demo. To run the current version with full location options, place the existing project dataset at:

```text
data/processed/hostwise_5cities.csv
```

This is a **local development/demo setup**, not yet a verified standalone deployment after a fresh clone. A future update can replace the full-dataset dependency with a small precomputed options file. The app's Python source also contains older `app/` path examples; the correct command for this repository layout is the `streamlit run streamlit/streamlit_app.py` command above.

## Limitations

- Predictions describe **historical advertised prices**, not transaction outcomes, occupancy guarantees, or verified fair market values.
- Unusual asking prices and listings with no reviews can produce greater uncertainty.
- Details such as property condition, photography quality, and exact location are not fully captured.
- The modeled dataset focuses on short-stay listings (`minimum_nights <= 29`) in the five listed cities.
- The fixed GBP/EUR conversion is a research convention, **not** a live exchange rate.

## Team

- Mohammed Almalki
- Rawan Alharibi
- Khalid Alshamri
- Raghad Alomari

**Saudi Digital Academy (SDA) × Le Wagon — Tabular Machine Learning Capstone, October 2026**

## Data Credit

Dataset: [Inside Airbnb](https://insideairbnb.com/). Please consult the original source for dataset documentation and terms of use.
