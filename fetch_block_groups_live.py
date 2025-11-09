"""
Fetch Philadelphia Census Block Group boundaries and ACS demographics using a live API key.
Outputs:
  - philadelphia_block_groups.geojson (geometries)
  - demographics_block_groups.csv (metrics used by the app)

ENV:
  CENSUS_API_KEY in environment or in a .env file (same folder)
"""

import os
import json
import pandas as pd
import geopandas as gpd
import requests
from typing import Dict, List

# Optional dotenv support (load both project root and MyKeys/.env if present)
def _load_env():
    try:
        from dotenv import load_dotenv
        # Try default .env in CWD
        load_dotenv()
        # Try MyKeys/.env relative to this file and to CWD
        candidates = [
            os.path.join(os.getcwd(), "MyKeys", ".env"),
            os.path.join(os.path.dirname(__file__), "MyKeys", ".env"),
        ]
        for p in candidates:
            if os.path.exists(p):
                load_dotenv(dotenv_path=p, override=False)
    except Exception:
        pass

_load_env()

STATE_FIPS = "42"   # Pennsylvania
COUNTY_FIPS = "101" # Philadelphia County
ACS_YEAR = "2022"
ACS_DATASET = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"

# Variables to fetch
# We'll use B01001 (Sex by Age) for more reliable age breakdowns
# Male: 007-011 (5-17), Female: 031-035 (5-17)
VAR_MAP: Dict[str, str] = {
    # Income & poverty
    "B19013_001E": "median_income",
    "B17001_001E": "pop_poverty_total",
    "B17001_002E": "pop_below_poverty",
    # K-12 age population (more reliable than B09001)
    # Male ages 5-17
    "B01001_007E": "male_5_9",
    "B01001_008E": "male_10_14",
    "B01001_009E": "male_15_17",
    # Female ages 5-17
    "B01001_031E": "female_5_9",
    "B01001_032E": "female_10_14",
    "B01001_033E": "female_15_17",
    # Total population
    "B01003_001E": "total_pop",
    # Race (selected)
    "B03002_003E": "white_alone",
    "B03002_004E": "black_alone",
    # Households with children under 18
    "B11005_002E": "hh_with_u18",
}


def _get_census_key() -> str | None:
    # Support several env var names
    for name in [
        "CENSUS_API_KEY",
        "CensusBureauAPI_KEY",
        "CENSUSBUREAUAPI_KEY",
        "CENSUS_KEY",
    ]:
        val = os.getenv(name)
        if val:
            return val
    return None


def fetch_acs_block_groups() -> pd.DataFrame:
    params = {
        "get": ",".join(VAR_MAP.keys()),
        "for": "block group:*",
        "in": f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
    }
    api_key = _get_census_key()
    if api_key:
        params["key"] = api_key

    r = requests.get(ACS_DATASET, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)

    # GEOID = state + county + tract + block_group
    df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]

    # Convert numeric fields
    for code, name in VAR_MAP.items():
        df[name] = pd.to_numeric(df[code], errors="coerce")

    # Derived fields - sum male and female ages 5-17
    df["k12_pop"] = (
        df["male_5_9"].fillna(0) + df["male_10_14"].fillna(0) + df["male_15_17"].fillna(0) +
        df["female_5_9"].fillna(0) + df["female_10_14"].fillna(0) + df["female_15_17"].fillna(0)
    )
    df["poverty_rate"] = (df["pop_below_poverty"] / df["pop_poverty_total"].replace(0, pd.NA)) * 100
    # Race percentages (if available)
    df["pct_black"] = (df["black_alone"] / df["total_pop"].replace(0, pd.NA)) * 100
    df["pct_white"] = (df["white_alone"] / df["total_pop"].replace(0, pd.NA)) * 100

    # Friendly schema for the app
    out = df[[
        "GEOID", "median_income", "k12_pop", "poverty_rate",
        "total_pop", "pct_black", "pct_white", "hh_with_u18"
    ]].copy()
    out.rename(columns={
        "GEOID": "block_group_id",
        "median_income": "income",
    }, inplace=True)

    return out


def fetch_block_group_shapes() -> gpd.GeoDataFrame:
    # 2022 TIGER/Line block groups for the state; filter to county
    url = f"https://www2.census.gov/geo/tiger/TIGER2022/BG/tl_2022_{STATE_FIPS}_bg.zip"
    gdf = gpd.read_file(url)
    gdf = gdf[gdf["COUNTYFP"] == COUNTY_FIPS].copy()
    gdf["GEOID"] = gdf["STATEFP"] + gdf["COUNTYFP"] + gdf["TRACTCE"] + gdf["BLKGRPCE"]
    gdf = gdf.to_crs(epsg=4326)
    return gdf


def main():
    print("Downloading block group shapes …")
    gdf = fetch_block_group_shapes()
    print(f"Shapes: {len(gdf)}")

    print("Fetching ACS data …")
    demo = fetch_acs_block_groups()
    print(f"ACS rows: {len(demo)}")

    # Merge and compute centroids
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    demo["block_group_id"] = demo["block_group_id"].astype(str)

    merged = gdf.merge(demo, left_on="GEOID", right_on="block_group_id", how="left")

    # Compute projected centroids for accurate lat/lon
    merged_proj = merged.to_crs(3857)
    cent_proj = merged_proj.geometry.centroid
    cent_ll = gpd.GeoSeries(cent_proj, crs=3857).to_crs(4326)
    merged["lat"] = cent_ll.y
    merged["lon"] = cent_ll.x
    # Ensure only one geometry column exists when writing
    if "centroid" in merged.columns:
        del merged["centroid"]

    # Add placeholders to match the app's expected columns
    merged["%Christian"] = 25.0
    merged["%first_gen"] = 35.0

    print("Saving outputs …")
    merged.to_file("philadelphia_block_groups.geojson", driver="GeoJSON")

    demographics = merged[[
        "block_group_id", "income", "k12_pop", "poverty_rate", "lat", "lon",
        "%Christian", "%first_gen", "total_pop", "pct_black", "pct_white", "hh_with_u18",
        "TRACTCE"
    ]].copy()
    demographics.to_csv("demographics_block_groups.csv", index=False)

    print("Done. Files written:\n - philadelphia_block_groups.geojson\n - demographics_block_groups.csv")


if __name__ == "__main__":
    main()
