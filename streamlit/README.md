# HostWise Streamlit app

**HostWise — Airbnb Price Prediction & Comparison.** Estimates a typical nightly price with the saved regression pipeline and, optionally, compares an asking price using `classify_price()` (Option C design).

## Run
From the project root:

```bash
pip install -r app/requirements.txt   # streamlit + the model libraries
python app/check_app.py               # quick check with the real model
streamlit run app/streamlit_app.py    # opens http://localhost:8501
```

## Files
| File | Purpose |
|---|---|
| `app/streamlit_app.py` | User interface (entry point) |
| `app/hostwise_service.py` | Loading, validation, prediction, usual range, comparison (no Streamlit) |
| `app/check_app.py` | Scenario check with the real pipeline |
| `.streamlit/config.toml` | Theme |

Uses, without modifying: `models/hostwise_pipeline.joblib`, `models/hostwise_pipeline_meta.json`, `models/pricing_thresholds.json`, `shared_foundation/preprocessing.py`, `shared_foundation/pricing_classifier.py`, and `data/processed/hostwise_5cities.csv` (only to list neighbourhoods, their typical coordinates and property types).
