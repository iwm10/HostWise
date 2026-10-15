# HostWise Streamlit app

**HostWise — Airbnb Price Prediction & Comparison.** Estimates a typical nightly price with the saved regression pipeline and, optionally, compares an asking price using `classify_price()` (Option C design).

## Run
From the project root, with Python 3.12:

```bash
pip install -r streamlit/requirements.txt   # streamlit + the model libraries (pinned)
python streamlit/check_app.py               # quick check with the real model
streamlit run streamlit/streamlit_app.py    # opens http://localhost:8501
```

## Deploy (Streamlit Community Cloud)
Repository `iwm10/HostWise`, branch `main`, main file path `streamlit/streamlit_app.py`, Python **3.12** (Advanced settings).
Streamlit Cloud installs `streamlit/requirements.txt` and the system package in `streamlit/packages.txt` (`libgomp1`, needed by LightGBM). No secrets are required.

## Files
| File | Purpose |
|---|---|
| `streamlit/streamlit_app.py` | User interface (entry point) |
| `streamlit/hostwise_service.py` | Loading, validation, prediction, usual range, comparison (no Streamlit) |
| `streamlit/app_options.json` | Neighbourhoods with typical coordinates, city bounds and property types per room type (derived from the training data) |
| `streamlit/build_app_options.py` | Regenerates `app_options.json` from `data/processed/hostwise_5cities.csv` (offline, only if the data changes) |
| `streamlit/check_app.py` | Scenario check with the real pipeline |
| `streamlit/requirements.txt`, `streamlit/packages.txt` | Python and system dependencies |
| `.streamlit/config.toml` | Theme |

Uses, without modifying: `models/hostwise_pipeline.joblib`, `models/hostwise_pipeline_meta.json`, `models/pricing_thresholds.json`, `shared_foundation/preprocessing.py` and `shared_foundation/pricing_classifier.py`. The full dataset is **not** needed to run the app.
