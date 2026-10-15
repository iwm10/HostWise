"""HostWise — Airbnb Price Prediction & Comparison (Streamlit app).

Run from the project root:
    streamlit run streamlit/streamlit_app.py
"""
from __future__ import annotations

import html

import streamlit as st

import hostwise_service as svc

st.set_page_config(page_title="HostWise — Airbnb Price Prediction & Comparison", page_icon="🏠", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --ink:#14171f; --muted:#5b6170; --line:#e5e3dc; --brand:#0e7c7b; --brand-soft:#d7eeec; --bg:#f7f6f2;
        --amber:#b45309; --amber-soft:#fff4dc; --marker:#e8590c; }
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); }
.block-container { padding-top: 3.2rem; max-width: 1240px; }
h1, h2, h3, h4 { font-family: 'Inter', sans-serif; letter-spacing: -0.01em; color: var(--ink); }
.hw-header { display:flex; align-items:center; gap:14px; margin-bottom: 4px; }
.hw-logo { width:44px; height:44px; border-radius:12px; background: var(--brand); display:flex; align-items:center; justify-content:center; }
.hw-title { font-size: 1.65rem; font-weight: 800; color: var(--ink); line-height: 1.1; }
.hw-title span { color: var(--brand); }
.hw-sub { color: var(--muted); font-size: 0.98rem; margin: 2px 0 18px 58px; }
.hw-section { font-size: 0.78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--brand); margin-bottom: 2px; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: #ffffff; border-radius: 14px !important; border-color: var(--line) !important; }
.stButton > button[kind="primary"], .stFormSubmitButton > button { background: var(--brand); border-color: var(--brand); border-radius: 10px;
  font-weight: 600; padding: 0.6rem 1rem; }
.stButton > button[kind="primary"]:hover { background: #0b6665; border-color: #0b6665; }
.hw-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding: 22px 24px 18px; box-shadow: 0 2px 10px rgba(20,23,31,.05); }
.hw-label { color: var(--muted); font-size: .9rem; font-weight: 500; }
.hw-price { font-size: 2.9rem; font-weight: 800; color: var(--ink); line-height: 1.05; margin: 2px 0 2px; letter-spacing: -0.02em; }
.hw-price small { font-size: 1.05rem; font-weight: 500; color: var(--muted); letter-spacing: 0; }
.hw-gbp { color: var(--muted); font-size: .95rem; }
.hw-warn { background: var(--amber-soft); border: 1px solid #f3cf7a; color: #5a3b00; border-radius: 10px; padding: 10px 12px; margin: 12px 0 0; font-size: .9rem; }
.hw-warn b { color: var(--amber); }
.hw-note { background: #eef2f7; border: 1px solid #d5dde8; color: #2b3445; border-radius: 10px; padding: 9px 12px; margin: 10px 0 0; font-size: .86rem; }
.hw-range-title { color: var(--ink); font-weight: 600; font-size: .95rem; }
.hw-divider { border-top: 1px solid var(--line); margin: 18px 0 14px; }
.st-key-hw-result { padding: 22px 24px 16px !important; background: #fff; border-radius: 16px !important;
  box-shadow: 0 2px 10px rgba(20,23,31,.05); }
@media (max-width: 640px) { .st-key-hw-result { padding: 18px 16px 12px !important; } }
.st-key-hw-result [data-testid="stExpander"] li, .st-key-hw-result [data-testid="stExpander"] p { font-size: .86rem; line-height: 1.5; color: #363b45; }
.hw-stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-top: 18px; }
.hw-stat { background: var(--bg); border: 1px solid var(--line); border-radius: 12px; padding: 11px 14px; }
.hw-stat-label { color: var(--muted); font-size: .82rem; font-weight: 500; }
.hw-stat-value { color: var(--ink); font-size: 1.4rem; font-weight: 700; line-height: 1.25; white-space: nowrap; }
.hw-stat-value small { color: var(--muted); font-size: .85rem; font-weight: 500; }
.hw-hint { color: var(--muted); font-size: .9rem; margin-top: 16px; }
.hw-range-head { display:flex; justify-content:space-between; align-items:baseline; gap: 8px; color: var(--muted); font-size: .88rem; }
.hw-range-head b { color: var(--ink); font-weight: 600; white-space: nowrap; }
.hw-bar { position: relative; height: 62px; margin: 8px 8px 0; }
.hw-track { position:absolute; top: 24px; left:0; right:0; height: 8px; background:#ebe9e3; border-radius: 6px; }
.hw-band { position:absolute; top:0; height:8px; background: #8fd0cc; border-radius: 6px; }
.hw-est { position:absolute; top:-6px; width:2px; height:20px; background: var(--ink); margin-left:-1px; }
.hw-marker { position:absolute; top:-5px; width:16px; height:16px; margin-left:-8px; border-radius:50%; background: var(--marker); border: 3px solid #fff; box-shadow: 0 0 0 1.5px var(--marker); }
.hw-ask-label { position:absolute; top: 0; font-size:.78rem; font-weight:700; color: var(--marker); white-space:nowrap; }
.hw-bound { position:absolute; top: 40px; transform: translateX(-50%); font-size:.8rem; font-weight:600; color: #24504f; white-space:nowrap; }
.hw-legend { display:flex; flex-wrap:wrap; gap: 4px 16px; font-size: .76rem; color: var(--muted); margin: 2px 0 0; }
.hw-legend i { display:inline-block; vertical-align: middle; margin-right: 5px; }
.hw-legend .k-band { width: 14px; height: 8px; border-radius: 4px; background: #8fd0cc; }
.hw-legend .k-est { width: 2px; height: 12px; background: var(--ink); }
.hw-legend .k-ask { width: 9px; height: 9px; border-radius: 50%; background: var(--marker); }
.hw-chip { display:inline-flex; align-items:center; gap:6px; font-weight:600; font-size:.88rem; padding: 5px 12px; border-radius: 999px; margin-top: 16px; }
.chip-Typical { background:#e3f3f1; color:#0b5e5d; border:1px solid #a7d8d3; }
.chip-Above { background:#fdf0e6; color:#8a3c06; border:1px solid #f2c39c; }
.chip-Below { background:#eceff5; color:#33415c; border:1px solid #c7cfdd; }
.chip-Uncertain { background:#f0efeb; color:#4b4a45; border:1px solid #d6d4cc; }
.hw-msg { color:#363b45; font-size:.9rem; margin-top: 8px; }
.hw-foot { color:#7b808b; font-size:.8rem; margin-top: 16px; line-height: 1.45; }
.hw-empty { color: var(--muted); font-size: .95rem; line-height: 1.6; }
.hw-step { display:flex; gap:10px; margin: 10px 0; }
@media (max-width: 640px) { .hw-sub { margin-left: 0; } .hw-price { font-size: 2.3rem; } .hw-stat-value { font-size: 1.25rem; } }
[data-testid="stHorizontalBlock"] { align-items: flex-start; }
@media (min-width: 641px) { [data-testid="stColumn"]:has(.st-key-hw-results) { position: sticky; top: 3.2rem; } }
.hw-step b { background: var(--brand-soft); color: var(--brand); border-radius: 50%; width: 26px; height: 26px; min-width: 26px;
  display:flex; align-items:center; justify-content:center; font-size: .85rem; }
</style>
""", unsafe_allow_html=True)

LOGO = ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M3 11.5 12 4l9 7.5" stroke="#fff" '
        'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/><path d="M6 10v9h12v-9" stroke="#fff" '
        'stroke-width="2.2" stroke-linejoin="round"/><path d="M10 19v-4.5h4V19" stroke="#fff" stroke-width="2"/></svg>')
st.markdown(f'<div class="hw-header"><div class="hw-logo">{LOGO}</div><div class="hw-title">Host<span>Wise</span> '
            f'<span style="color:#14171f;font-weight:600;font-size:1.15rem">— Airbnb Price Prediction &amp; Comparison</span>'
            f'</div></div><div class="hw-sub">Estimate a typical nightly price for your listing in London, Paris, Rome, '
            f'Madrid or Barcelona, and see how your own price compares.</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading the HostWise model…")
def get_artifacts():
    return svc.load_artifacts()


@st.cache_data(show_spinner=False)
def get_options(_artifacts):
    return svc.listing_options(_artifacts)


try:
    artifacts = get_artifacts()
except Exception as exc:  # friendly message instead of a traceback
    st.error("The HostWise model could not be loaded. Make sure `models/hostwise_pipeline.joblib`, "
             "`models/hostwise_pipeline_meta.json`, `models/pricing_thresholds.json` and `shared_foundation/` "
             f"are present, and that the required libraries are installed.\n\nDetails: {exc}")
    st.stop()
options = get_options(artifacts)
GBP_RATE = artifacts.meta.get("gbp_to_eur", 1.15)


def section(title: str):
    st.markdown(f'<div class="hw-section">{title}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ input form
left, right = st.columns([1.08, 1], gap="large")

with left:
    with st.container(border=True):
        section("Location")
        c1, c2 = st.columns(2)
        city = c1.selectbox("City", list(svc.CITIES), index=1, format_func=svc.CITIES.get)
        city_opts = options["cities"].get(city)
        if city_opts:
            hoods = list(city_opts["neighbourhoods"])
            hood = c2.selectbox("Neighbourhood", hoods, help="Official Inside Airbnb neighbourhood names.")
            default = city_opts["neighbourhoods"][hood]
        else:
            hood = c2.text_input("Neighbourhood", placeholder="e.g. Hôtel-de-Ville")
            default = {"lat": None, "lon": None}
        with st.expander("Exact location (optional)"):
            st.caption("Defaults to the typical position of listings in the selected neighbourhood.")
            g1, g2 = st.columns(2)
            lat = g1.number_input("Latitude", value=default["lat"], format="%.5f", step=0.001, key=f"lat-{city}-{hood}")
            lon = g2.number_input("Longitude", value=default["lon"], format="%.5f", step=0.001, key=f"lon-{city}-{hood}")

    with st.container(border=True):
        section("Property")
        c1, c2 = st.columns(2)
        room_type = c1.selectbox("Room type", svc.ROOM_TYPES)
        property_type = c2.selectbox("Property type", options["property_types"][room_type])
        c1, c2, c3 = st.columns(3)
        guests = c1.number_input("Guests", min_value=1, max_value=16, value=2, step=1)
        size_choices = ["Not specified", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        bedrooms = c2.selectbox("Bedrooms", size_choices, index=2,
                                format_func=lambda v: "0 (studio)" if v == 0 else str(v))
        beds = c3.selectbox("Beds", size_choices[2:] + ["Not specified"], index=0)
        c1, c2 = st.columns(2)
        bath_count = c1.number_input("Number of bathrooms", min_value=1, max_value=10, value=1, step=1)
        bath_kind = c2.selectbox("Bathroom privacy", ["Not specified", "Private", "Shared"],
                                 help="Private or shared bathroom, as written on Airbnb (e.g. '1 shared bath'). "
                                      "Leave as Not specified for a whole home.")

    with st.container(border=True):
        section("Amenities")
        st.caption("Tick what the listing offers. These are the amenities the model uses.")
        cols = st.columns(2)
        chosen = [name for i, name in enumerate(svc.MODEL_AMENITIES)
                  if cols[i % 2].checkbox(name, key=f"amenity-{name}")]

    with st.container(border=True):
        section("Reviews & host")
        has_reviews = st.toggle("This listing already has reviews", value=True)
        reviews, rating = 0, None
        if has_reviews:
            c1, c2 = st.columns(2)
            reviews = c1.number_input("Number of reviews", min_value=1, max_value=5000, value=25, step=1)
            rating = c2.slider("Average rating", min_value=1.0, max_value=5.0, value=4.8, step=0.05)
        else:
            st.caption("New listing: the model uses what it learned from listings without reviews.")
        superhost = st.checkbox("Host is a Superhost")
        with st.expander("More host and stay details (optional)"):
            c1, c2, c3 = st.columns(3)
            years = c1.number_input("Years hosting", min_value=0, max_value=20, value=2, step=1)
            host_listings = c2.number_input("Listings managed", min_value=1, max_value=1000, value=1, step=1)
            min_nights = c3.number_input("Minimum nights", min_value=1, max_value=29, value=2, step=1)

    with st.container(border=True):
        section("Your asking price (optional)")
        asking = st.number_input("Proposed nightly price in EUR", min_value=1.0, max_value=100000.0, value=None,
                                 step=5.0, placeholder="e.g. 420", help="Leave empty to see the estimate only.")
        if city == "london":
            st.caption(f"London: enter the price in EUR (1 GBP = {GBP_RATE} EUR"
                       + (f"; €{asking:,.0f} ≈ £{asking / GBP_RATE:,.0f})." if asking else ")."))

    submitted = st.button("Estimate nightly price", type="primary", use_container_width=True)

listing = {
    "city": city, "neighbourhood_cleansed": hood or None, "latitude": lat, "longitude": lon,
    "room_type": room_type, "property_type": property_type, "accommodates": int(guests),
    "bedrooms": None if bedrooms == "Not specified" else float(bedrooms),
    "beds": None if beds == "Not specified" else float(beds),
    "bathrooms_text": svc.bathrooms_text(int(bath_count), bath_kind),
    "minimum_nights": float(min_nights), "number_of_reviews": int(reviews), "review_scores_rating": rating,
    "host_is_superhost": "t" if superhost else "f", "calculated_host_listings_count": int(host_listings),
    "hosts_time_as_host_years": float(years), "amenities": svc.amenities_text(chosen),
}
snapshot = (tuple(sorted((k, str(v)) for k, v in listing.items())), asking)

if submitted:
    errors = svc.validate(listing, options)
    if errors:
        st.session_state.pop("result", None)
        st.session_state["errors"] = errors
    else:
        st.session_state.pop("errors", None)
        try:
            st.session_state["result"] = svc.assess(artifacts, listing, asking)
            st.session_state["snapshot"] = snapshot
            st.session_state["asking"] = asking
        except ValueError as exc:
            st.session_state["errors"] = [f"Please check your inputs: {exc}"]
        except Exception as exc:  # unexpected model error, shown politely
            st.session_state["errors"] = [f"Something went wrong while estimating the price ({exc})."]


# ------------------------------------------------------------------ result rendering
def eur(v: float) -> str:
    return f"€{v:,.0f}"


def range_bar(estimate: float, rng: dict, asking: float | None) -> str:
    """Track with the usual band, an estimate tick and the asking-price marker.

    Rows never share space: asking-price label above the track, the two bounds below it,
    and the estimate is identified in a legend row underneath.
    """
    lo, hi = min(0.6 * estimate, rng["lower"] * 0.9), max(1.5 * estimate, rng["upper"] * 1.1)
    if asking is not None:  # widen moderately so the marker usually sits on the track
        lo, hi = min(lo, max(asking * 0.95, 0.4 * estimate)), max(hi, min(asking * 1.05, 2.2 * estimate))
    pos = lambda v: max(0.0, min(100.0, 100 * (v - lo) / (hi - lo)))
    track = [f'<div class="hw-band" style="left:{pos(rng["lower"]):.2f}%;width:{pos(rng["upper"]) - pos(rng["lower"]):.2f}%"></div>',
             f'<div class="hw-est" style="left:{pos(estimate):.2f}%"></div>']
    above = []
    below = [f'<div class="hw-bound" style="left:{pos(rng["lower"]):.2f}%">{eur(svc.round_to_5(rng["lower"]))}</div>',
             f'<div class="hw-bound" style="left:{pos(rng["upper"]):.2f}%">{eur(svc.round_to_5(rng["upper"]))}</div>']
    legend = ['<span><i class="k-band"></i>Usual observed range</span>',
              f'<span><i class="k-est"></i>Estimate {eur(estimate)}</span>']
    if asking is not None:
        p = pos(asking)
        text = f"Your price {eur(asking)}"
        text = f"{text} ▶" if asking > hi else (f"◀ {text}" if asking < lo else text)
        anchor = "left:0" if p < 20 else ("right:0" if p > 80 else f"left:{p:.2f}%;transform:translateX(-50%)")
        track.append(f'<div class="hw-marker" style="left:{p:.2f}%"></div>')
        above.append(f'<div class="hw-ask-label" style="{anchor}">{text}</div>')
        legend.append('<span><i class="k-ask"></i>Your asking price</span>')
    return (f'<div class="hw-bar">{"".join(above)}<div class="hw-track">{"".join(track)}</div>{"".join(below)}</div>'
            f'<div class="hw-legend">{"".join(legend)}</div>')


def result_html(r: svc.Result, asking: float | None) -> str:
    """Option C order: estimate, asking-price comparison, usual range, status, warnings, disclaimer."""
    # A. Estimated nightly price (headline rounded to 5 EUR; everything else uses the precise estimate)
    out = ['<div class="hw-label">Estimated nightly price</div>',
           f'<div class="hw-price">About {eur(r.estimate_display)} <small>/ night</small></div>']
    if r.city == "london":
        out.append(f'<div class="hw-gbp">≈ £{svc.round_to_5(r.estimate_eur / GBP_RATE):,} per night '
                   f'(1 GBP = {GBP_RATE} EUR, fixed rate)</div>')

    # B. Asking price comparison
    c = r.comparison
    if c:
        sign = "+" if c["difference_eur"] >= 0 else "−"
        out.append('<div class="hw-stats">'
                   f'<div class="hw-stat"><div class="hw-stat-label">Your asking price</div>'
                   f'<div class="hw-stat-value">{eur(c["listed_price_eur"])}<small> / night</small></div></div>'
                   f'<div class="hw-stat"><div class="hw-stat-label">Difference from estimate</div>'
                   f'<div class="hw-stat-value">{sign}{eur(abs(c["difference_eur"]))} '
                   f'<small>({sign}{abs(c["difference_percent"]):.1f}%)</small></div></div></div>')
    else:
        out.append('<div class="hw-hint">Add an asking price on the left to compare it with the estimate.</div>')

    # C. Usual observed range (secondary)
    out.append('<div class="hw-divider"></div>')
    if r.usual_range:
        lower, upper = (svc.round_to_5(r.usual_range[k]) for k in ("lower", "upper"))
        out.append(f'<div class="hw-range-head"><span>Usual observed range</span><b>{eur(lower)}–{eur(upper)}</b></div>')
        out.append(range_bar(r.estimate_eur, r.usual_range, asking if c else None))
    else:
        out.append('<div class="hw-range-head"><span>Usual observed range</span><b>Not shown</b></div>')

    # D. Pricing status from classify_price()
    if c:
        status = c["pricing_status"]
        icon = {"Typical": "●", "Above Typical": "▲", "Below Typical": "▼", "Uncertain": "?"}[status]
        out.append(f'<div class="hw-chip chip-{status.split()[0]}">{icon} {svc.STATUS_LABELS[status]}</div>')
        out.append(f'<div class="hw-msg">{svc.STATUS_MESSAGES[status]}</div>')

    # E. Warnings stay visible; methodology lives in the expander below
    for w in r.warnings:
        out.append(f'<div class="hw-warn"><b>⚠ Please note</b><br>{html.escape(w)}</div>')
    for n in r.notes:
        out.append(f'<div class="hw-note">ℹ {html.escape(n)}</div>')
    out.append('<div class="hw-foot">Based on historical listing prices. Not a guaranteed fair-market valuation.</div>')
    return "".join(out)


def methodology_md(r: svc.Result) -> str:
    m, t = artifacts.meta, artifacts.thresholds
    seg = t["segments"]["with_reviews" if r.has_reviews else "no_reviews"]
    group = "listings with reviews" if r.has_reviews else "listings without reviews"
    city_mae = m.get("cv_mae_by_city_eur", {}).get(r.city)
    lines = [
        f"- **Estimate:** {m.get('model', 'the saved model')}, trained on {m.get('n_train', 0):,} listings. "
        f"Precise model output for this listing: **€{r.estimate_eur:,.2f}**; the headline is rounded to the nearest €5.",
        f"- **Usual observed range:** where the middle 50% of comparable {group} were priced relative to their "
        f"model estimate (−{(1 - seg['typical_low']) * 100:.0f}% to +{(seg['typical_high'] - 1) * 100:.0f}% of the "
        "estimate), calibrated on out-of-fold training predictions. It reflects historical listing-price variation, "
        "not a verified fair market value or a guaranteed prediction interval.",
        "- **Pricing status:** your asking price is compared with that range. Prices far outside what similar listings "
        "charged, or estimates above "
        f"€{t['max_reliable_estimate_eur']:,}, are marked *Can't assess reliably*.",
    ]
    if city_mae:
        lines.append(f"- **Typical error:** cross-validated mean absolute error in {svc.CITIES[r.city]} is about "
                     f"€{city_mae:,.0f}" + ("" if r.has_reviews else
                     f"; for listings without reviews it is about €{m.get('cv_mae_no_reviews_eur', 0):,.0f}") + ".")
    if r.city == "london":
        lines.append(f"- **London:** the model works in EUR; GBP amounts use a fixed rate of 1 GBP = {GBP_RATE} EUR.")
    return "\n".join(lines)


with right, st.container(key="hw-results"):
    if st.session_state.get("errors"):
        st.error("Please fix the following before estimating:\n\n" +
                 "\n".join(f"- {e}" for e in st.session_state["errors"]))
    result = st.session_state.get("result")
    if result is None:
        st.markdown(
            '<div class="hw-card"><div class="hw-range-title" style="font-size:1.1rem">How it works</div>'
            '<div class="hw-step"><b>1</b><div class="hw-empty">Describe the listing: location, property, amenities and reviews.</div></div>'
            '<div class="hw-step"><b>2</b><div class="hw-empty">Optionally add the nightly price you plan to ask.</div></div>'
            '<div class="hw-step"><b>3</b><div class="hw-empty">Get a typical nightly price and see where your price sits '
            'compared with similar listings.</div></div></div>', unsafe_allow_html=True)
    else:
        if st.session_state.get("snapshot") != snapshot:
            st.info("You changed some details. Press **Estimate nightly price** to update the result.")
        with st.container(border=True, key="hw-result"):
            st.markdown(result_html(result, st.session_state.get("asking")), unsafe_allow_html=True)
            with st.expander("How is this calculated?"):
                st.markdown(methodology_md(result))

    with st.expander("About the model"):
        m = artifacts.meta
        st.markdown(
            f"- **Model:** {m.get('model', '—')}, preprocessing {m.get('preprocessing', '—')}\n"
            f"- **Cross-validation MAE:** €{m.get('cv_mae_eur', float('nan')):,.2f}\n"
            f"- **Held-out test MAE (official):** €{m.get('test_mae_eur', float('nan')):,.2f} — "
            f"median absolute error €{m.get('test_median_ae_eur', float('nan')):,.2f}\n"
            f"- **Trained:** {m.get('trained_date', '—')} on {m.get('n_train', 0):,} listings\n"
            "- Prices are nightly asking prices from Inside Airbnb listings (London converted at 1 GBP = 1.15 EUR). "
            "The estimate is a typical asking price for similar listings, not a guaranteed market value.")
        if artifacts.version_mismatch:
            st.warning("Installed library versions differ from those used to train the model: "
                       + ", ".join(f"{k} {v['installed']} (trained {v['trained']})"
                                   for k, v in artifacts.version_mismatch.items()))
