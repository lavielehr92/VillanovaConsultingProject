#!/usr/bin/env python3
"""
Build an interactive Google Maps competitors map for Cornerstone Christian Academy (Philadelphia).
- Reads competitors from a CSV (name,address,city,state,zip,type,website,notes).
- Geocodes addresses with Google Geocoding API.
- Caches results to avoid repeat lookups.
- Writes an HTML map (Google Maps JavaScript API) with clustered markers and info windows.

USAGE
-----
1) Put your competitors in competitors.csv (see the sample file created with this script).
2) Set your Google API key in the environment as GOOGLE_MAPS_API_KEY (recommended) or pass with --api-key.
3) Run:
   python map_competitors.py --csv competitors.csv --out competitors_map.html

REQUIREMENTS
------------
- Python 3.8+
- pip install requests pandas

Billable Note: Geocoding and Maps JS API calls may incur Google Cloud charges. Use quotas and caching.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# --------- Config (edit if desired) ---------
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
DEFAULT_CENTER = (39.9526, -75.1652)  # Philadelphia City Hall lat/lng
DEFAULT_ZOOM = 11
RATE_LIMIT_SECONDS = 0.15  # sleep between geocode calls to be gentle
CACHE_FILE = "geocode_cache.json"
# --------------------------------------------

def load_cache(cache_path: Path) -> Dict[str, Dict]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(cache_path: Path, cache: Dict[str, Dict]) -> None:
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def geocode(address: str, api_key: str, cache: Dict[str, Dict]) -> Optional[Tuple[float, float, Dict]]:
    """Return (lat, lng, full_result) or None if not found."""
    key = address.strip().lower()
    if key in cache:
        data = cache[key]
        return data.get("lat"), data.get("lng"), data.get("result")
    params = {"address": address, "key": api_key}
    r = requests.get(GEOCODE_URL, params=params, timeout=20)
    if r.status_code != 200:
        print(f"[WARN] HTTP {r.status_code} for address: {address}")
        return None
    payload = r.json()
    status = payload.get("status")
    if status != "OK" or not payload.get("results"):
        print(f"[WARN] Geocode failed for '{address}': status={status}")
        return None
    best = payload["results"][0]
    loc = best["geometry"]["location"]
    lat, lng = loc["lat"], loc["lng"]
    cache[key] = {"lat": lat, "lng": lng, "result": best}
    return lat, lng, best

def df_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize columns
    expected = ["name","address","city","state","zip","type","website","notes"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    return df[expected]

def make_full_address(row) -> str:
    bits = [row.get("address",""), row.get("city",""), row.get("state",""), str(row.get("zip",""))]
    return ", ".join([b for b in bits if str(b).strip()])

def build_markers(df: pd.DataFrame) -> List[Dict]:
    markers = []
    for _, r in df.iterrows():
        markers.append({
            "name": r["name"],
            "type": r["type"],
            "website": r["website"],
            "notes": r["notes"],
            "address_display": make_full_address(r),
            "lat": float(r["lat"]),
            "lng": float(r["lng"]),
        })
    return markers

def write_html(out_path: Path, api_key: str, markers: List[Dict], center: Tuple[float,float]=DEFAULT_CENTER, zoom: int=DEFAULT_ZOOM):
    # Embed markers as JSON
    markers_json = json.dumps(markers, ensure_ascii=False)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Competitors Map — Cornerstone Christian Academy</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
    .infowin h3 {{ margin: 0 0 4px 0; font-size: 16px; }}
    .infowin p {{ margin: 0; }}
    .legend {{ background: white; padding: 8px 10px; border: 1px solid #ddd; border-radius: 6px; position: absolute; bottom: 20px; left: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.2); }}
    .legend b {{ display:block; margin-bottom: 6px; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="legend">
    <b>Legend</b>
    <div>Blue pins: Competitors</div>
  </div>

  <script>
    const MARKERS = {markers_json};

    function initMap() {{
      const map = new google.maps.Map(document.getElementById('map'), {{
        center: {{lat: {center[0]}, lng: {center[1]}}},
        zoom: {zoom},
        mapTypeControl: false,
      }});

      const info = new google.maps.InfoWindow();

      const markers = MARKERS.map(m => {{
        const marker = new google.maps.Marker({{
          position: {{lat: m.lat, lng: m.lng}},
          title: m.name
        }});
        const html = `
          <div class="infowin">
            <h3>${{m.name}}</h3>
            <p><b>Type:</b> ${{m.type || '—'}}</p>
            <p><b>Address:</b> ${{m.address_display || '—'}}</p>
            ${{m.website ? `<p><a href="${{m.website}}" target="_blank" rel="noopener">Website</a></p>` : ''}}
            ${{m.notes ? `<p>${{m.notes}}</p>` : ''}}
          </div>`;
        marker.addListener('click', () => {{
          info.setContent(html);
          info.open(map, marker);
        }});
        return marker;
      }});

      // Marker clustering
      new markerClusterer.MarkerClusterer({{
        map,
        markers,
      }});
    }}
  </script>
  <!-- MarkerClusterer (official Google library) -->
  <script src="https://unpkg.com/@googlemaps/markerclusterer/dist/index.min.js"></script>
  <!-- Google Maps JS API (insert your key) -->
  <script async src="https://maps.googleapis.com/maps/api/js?key=%s&callback=initMap"></script>
</body>
</html>""" % api_key

    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote map HTML: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Build competitors map (Google Maps).")
    parser.add_argument("--csv", type=str, default="competitors.csv", help="Input CSV with competitors")
    parser.add_argument("--out", type=str, default="competitors_map.html", help="Output HTML file")
    parser.add_argument("--api-key", type=str, default=os.environ.get("GOOGLE_MAPS_API_KEY","YOUR_GOOGLE_MAPS_API_KEY"),
                        help="Google Maps API key (or set env GOOGLE_MAPS_API_KEY)")
    parser.add_argument("--center", type=str, default=f"{DEFAULT_CENTER[0]},{DEFAULT_CENTER[1]}", help="Center lat,lng")
    parser.add_argument("--zoom", type=int, default=DEFAULT_ZOOM, help="Initial zoom")
    parser.add_argument("--skip-geocode", action="store_true", help="Skip geocoding (assumes lat/lng already in CSV)")
    args = parser.parse_args()

    api_key = args.api_key.strip()
    in_path = Path(args.csv)
    out_path = Path(args.out)
    cache_path = Path(CACHE_FILE)

    if not in_path.exists():
        raise SystemExit(f"Input CSV not found: {in_path}")

    df = df_from_csv(in_path)

    if not args.skip_geocode:
        cache = load_cache(cache_path)
        lats = []
        lngs = []
        for i, row in df.iterrows():
            full_address = make_full_address(row)
            # Fallback to 'address' alone if others empty
            q = full_address if full_address else row["address"]
            if not q:
                lats.append(float('nan')); lngs.append(float('nan'))
                print(f"[WARN] Row {i} missing address: {row.to_dict()}")
                continue
            res = geocode(q, api_key, cache)
            if not res:
                lats.append(float('nan')); lngs.append(float('nan'))
            else:
                lat, lng, _ = res
                lats.append(lat); lngs.append(lng)
            time.sleep(RATE_LIMIT_SECONDS)
        save_cache(cache_path, cache)
        df["lat"] = lats
        df["lng"] = lngs
        # Persist geocoded CSV
        geo_csv = in_path.with_name(in_path.stem + "_geocoded.csv")
        df.to_csv(geo_csv, index=False)
        print(f"[OK] Wrote geocoded CSV: {geo_csv}")
    else:
        # Expect lat/lng columns already present
        missing = [c for c in ("lat","lng") if c not in df.columns]
        if missing:
            raise SystemExit(f"--skip-geocode set, but CSV missing columns: {missing}")

    # Drop rows without coords
    df = df.dropna(subset=["lat","lng"])

    # Compute center if at least one marker
    center_lat, center_lng = DEFAULT_CENTER
    if len(df) > 0:
        center_lat = float(df["lat"].mean())
        center_lng = float(df["lng"].mean())

    markers = build_markers(df)
    write_html(out_path, api_key, markers, center=(center_lat, center_lng), zoom=args.zoom)

if __name__ == "__main__":
    main()
