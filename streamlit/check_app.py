"""Quick end-to-end check of the HostWise app logic with the real saved pipeline.

Run from the project root:  python app/check_app.py
"""
import hostwise_service as svc

artifacts = svc.load_artifacts()
options = svc.listing_options(artifacts)
print(f"Model loaded: {artifacts.meta.get('model')} | inputs: {len(artifacts.inputs)} | "
      f"version mismatches: {artifacts.version_mismatch or 'none'}")

paris = {
    "city": "paris", "neighbourhood_cleansed": "Hôtel-de-Ville", "room_type": "Entire home/apt",
    "property_type": "Entire rental unit", "accommodates": 4, "bedrooms": 2, "beds": 2,
    "bathrooms_text": "1 bath", "latitude": 48.8566, "longitude": 2.3522, "minimum_nights": 2,
    "review_scores_rating": 4.8, "number_of_reviews": 25, "host_is_superhost": "f",
    "calculated_host_listings_count": 1, "hosts_time_as_host_years": 3,
    "amenities": svc.amenities_text(["Washer", "Elevator", "Dedicated workspace"]),
}
london = {**paris, "city": "london", "neighbourhood_cleansed": "Westminster", "latitude": 51.4975, "longitude": -0.1357}

cases = [
    ("Paris, with reviews, no asking price", paris, None),
    ("Paris, with reviews, asking 420", paris, 420),
    ("Same listing without reviews, asking 374", {**paris, "number_of_reviews": 0}, 374),
    ("London (GBP display), no asking price", london, None),
    ("Unusual property type", {**paris, "property_type": "Treehouse"}, None),
    ("Asking price typo (x10)", paris, 4000),
]
for name, listing, asking in cases:
    errors = svc.validate(listing, options)
    if errors:
        print(f"\n{name}: VALIDATION ERRORS {errors}")
        continue
    r = svc.assess(artifacts, listing, asking)
    status = r.comparison["pricing_status"] if r.comparison else "—"
    gbp = f" (≈ £{r.estimate_eur / artifacts.meta.get('gbp_to_eur', 1.15):,.0f})" if r.city == "london" else ""
    print(f"\n{name}\n  estimate €{r.estimate_eur:,.2f} → about €{r.estimate_display}{gbp}\n"
          f"  usual range {r.usual_range}\n  status {status}\n  warnings {r.warnings}\n  notes {r.notes}")

print("\nInvalid input check:", svc.validate({**paris, "neighbourhood_cleansed": "", "latitude": 40.0}, options))
