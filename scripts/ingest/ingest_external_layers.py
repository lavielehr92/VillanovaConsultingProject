"""Ingest external spatial datasets and aggregate to Philadelphia block-groups.

Outputs CSVs to data/external/ with the expected schema consumed by the
Streamlit app. The script tries online sources where reasonable but will
fall back to local files if present.

This is intentionally conservative: fetches are optional and failure modes
fall back to writing CSVs that will be ignored by the app if empty.

Usage: python scripts/ingest/ingest_external_layers.py --help

CLI flags:
    --census-key <key>          : Census API key for gini fetch (optional)
    --crime-endpoint <url>      : Custom crime dataset endpoint (Socrata/ArcGIS CSV/JSON)
    --vacancy-endpoint <url>    : Custom vacant properties endpoint
    --transit-endpoint <url>    : Custom transit stops endpoint
    --food-endpoint <url>       : Custom food store endpoint
    --hud-endpoint <url>        : Custom HUD assisted property endpoint
    --raw-dir <path>            : Directory for local raw fallback files

Supported layers:
- crime -> crime_per_block_group.csv (columns: block_group_id, crimes)
- vacancy -> vacancy_block_groups.csv (columns: block_group_id, vacant_count, vacant_pct)
- transit -> transit_access_block_groups.csv (columns: block_group_id, transit_stop_count, transit_access_score)
- food -> food_access_block_groups.csv (columns: block_group_id, grocery_count, food_access_score)
- gini -> gini_block_groups.csv (columns: block_group_id, gini_index)
- hud -> hud_assisted_block_groups.csv (columns: block_group_id, hud_assisted)

"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus
from datetime import datetime, timedelta

try:
    import geopandas as gpd
except Exception:
    gpd = None
import numpy as np
import pandas as pd
import io
import requests
try:
    from shapely.geometry import Point
except Exception:
    Point = None

# Paths used by app_block_groups
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_EXTERNAL = ROOT / "data" / "external"
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

# Path to local block-group GeoJSON (expected by app)
BG_GEOJSON = ROOT / "philadelphia_block_groups.geojson"

# Default input fallbacks if downloads fail
FALLBACK_DIR = ROOT / "data" / "raw"

# Common headers for outputs


def load_block_group_gdf() -> gpd.GeoDataFrame:
    if not BG_GEOJSON.exists():
        raise RuntimeError(f"Block-group GeoJSON not found at {BG_GEOJSON}. Run fetch_block_groups.py first.")
    gdf = gpd.read_file(BG_GEOJSON)
    # Normalize expected id
    if 'GEOID' in gdf.columns:
        gdf['block_group_id'] = gdf['GEOID'].astype(str)
    else:
        raise RuntimeError("Loaded block-group GeoJSON missing 'GEOID' column")
    # Ensure WGS84 CRS
    if gdf.crs is None or gdf.crs.to_string() != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)
    return gdf[['block_group_id', 'geometry']]


# ---------------------------------------------------------------------------
# Generic point aggregation helper
# ---------------------------------------------------------------------------

def aggregate_points_to_bg(points: gpd.GeoDataFrame, bg_gdf: gpd.GeoDataFrame, value_col: Optional[str] = None, agg_name: str = "count") -> pd.DataFrame:
    """Spatial join points onto block-groups and aggregate.

    Returns frames with index of block_group_id and aggregated columns: 
    - count if value_col is None (number of points)
    - sum(value_col) if provided
    """
    if points.empty:
        return pd.DataFrame(columns=['block_group_id', agg_name])

    if points.crs is None:
        points = points.set_crs(epsg=4326)
    else:
        points = points.to_crs(epsg=4326)

    joined = gpd.sjoin(bg_gdf, points, how='left', predicate='intersects')
    if value_col and value_col in joined.columns:
        agg = joined.groupby('block_group_id')[value_col].sum().reset_index()
        agg = agg.rename(columns={value_col: agg_name})
    else:
        agg = joined.groupby('block_group_id').size().reset_index(name=agg_name)
    return agg


# ---------------------------------------------------------------------------
# Crime
# ---------------------------------------------------------------------------

def build_cartosql_url(sql: str, fmt: str = 'geojson') -> str:
    base = 'https://phl.carto.com/api/v2/sql'
    return f"{base}?format={fmt}&q={quote_plus(sql)}"


def fetch_philly_crime_data(start: Optional[str] = None, end: Optional[str] = None, fmt: str = 'geojson', endpoint: Optional[str] = None, raw_path: Path | None = None) -> gpd.GeoDataFrame:
    """Return GeoDataFrame of crime incidents with lat/lon columns incl 'geometry'.

    Tries to fetch Philly's crime data (Socrata endpoint). If the network fetch
    fails or the endpoint changes, the function falls back to a local CSV in
    data/raw/crime_incidents.csv.
    """
    # Best guess Socrata dataset for Philadelphia crimes (subject to change).
    CARTO_BASE = endpoint or "https://phl.carto.com/api/v2/sql"

    raw_path = raw_path or (FALLBACK_DIR / 'crime_incidents.csv')

    df = None
    # Create SQL using user provided start & end; default to last 30 days
    if start is None or end is None:
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=30)
        start = start or start_dt.strftime('%Y-%m-%d %H:%M:%S')
        end = end or end_dt.strftime('%Y-%m-%d %H:%M:%S')

    # SQL Template
    sql = (
        "SELECT *, ST_Y(the_geom) AS lat, ST_X(the_geom) AS lon "
        "FROM incidents_part1_part2 "
        f"WHERE dispatch_date_time >= '{start}' "
        f"AND dispatch_date_time < '{end}'"
    )
    try:
        carto_url = build_cartosql_url(sql, fmt=fmt)
        resp = requests.get(carto_url, timeout=60)
        resp.raise_for_status()
        if fmt == 'geojson':
            geojson = resp.json()
            # parse features into DataFrame
            rows = [f.get('properties', {}) for f in geojson.get('features', [])]
            df = pd.DataFrame.from_records(rows)
            # lat/lon may not be present in properties if they are in geometry but we added ST_Y/ST_X
            if 'lat' not in df.columns and 'the_geom' in df.columns:
                # try to extract
                try:
                    df['lat'] = df['the_geom'].apply(lambda g: g.get('y'))
                    df['lon'] = df['the_geom'].apply(lambda g: g.get('x'))
                except Exception:
                    pass
        else:
            # csv
            df = pd.read_csv(io.StringIO(resp.text))
    except Exception:  # pragma: no cover - remote source may not be available in test env
        df = None

    if df is None and raw_path.exists():
        df = pd.read_csv(raw_path)

    if df is None:
        # Return an empty gdf, the app will handle defaults
        return gpd.GeoDataFrame(columns=['geometry'])

    # Normalize lat/lon and geometry
    lat_col, lon_col = None, None
    for candidate in ('lat', 'latitude', 'y', 'INTPTLAT'):
        if candidate in df.columns:
            lat_col = candidate
            break
    for candidate in ('lon', 'longitude', 'x', 'INTPTLON'):
        if candidate in df.columns:
            lon_col = candidate
            break

    if lat_col and lon_col:
        df['lat'] = pd.to_numeric(df[lat_col], errors='coerce')
        df['lon'] = pd.to_numeric(df[lon_col], errors='coerce')
        df = df.dropna(subset=['lat', 'lon'])
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326')
        return gdf
    else:
        return gpd.GeoDataFrame(columns=['geometry'])


# ---------------------------------------------------------------------------
# Vacancy
# ---------------------------------------------------------------------------

def fetch_vacant_properties(endpoint: Optional[str] = None, raw_path: Path | None = None) -> gpd.GeoDataFrame:
    """Fetch vacant properties dataset from the City of Philadelphia OpenData.

    Falls back to data/raw/vacant_properties.csv when the network isn't available.
    The script returns a geodataframe with geometry as points and the 'vacant'
    flag present if possible.
    """
    # Placeholder endpoint - will need to be adapted based on official Socrata/ArcGIS endpoint
    VACANT_ENDPOINT = endpoint or "https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/Vacant_Block_Percent_Combined/FeatureServer/0/query?where=1=1&outFields=*&f=geojson"

    raw_path = raw_path or (FALLBACK_DIR / 'vacant_properties.csv')

    df = None
    try:
        resp = requests.get(VACANT_ENDPOINT, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception:
        if raw_path.exists():
            df = pd.read_csv(raw_path)

    if df is None:
        return gpd.GeoDataFrame(columns=['geometry'])

    lat_col = next((c for c in df.columns if c.lower() in ('lat', 'latitude', 'y', 'intptlat')), None)
    lon_col = next((c for c in df.columns if c.lower() in ('lon', 'longitude', 'x', 'intptlon')), None)
    df['lat'] = pd.to_numeric(df[lat_col], errors='coerce') if lat_col else pd.NA
    df['lon'] = pd.to_numeric(df[lon_col], errors='coerce') if lon_col else pd.NA

    df = df.dropna(subset=['lat', 'lon'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326')
    # The original dataset may have a 'vacant' or 'vacancy' indicator; we'll set a count
    gdf['vacant'] = 1
    return gdf


# ---------------------------------------------------------------------------
# Transit
# ---------------------------------------------------------------------------

def fetch_transit_stops(endpoint: Optional[str] = None, raw_path: Path | None = None) -> gpd.GeoDataFrame:
    """Fetch SEPTA transit stops (bus/rail) if available and return as point GeoDataFrame.

    This function prefers local fallback if provided in data/raw/septa_stops.csv.
    The final metric used by the app is a simple stop count per block group and
    a normalized transit_access_score derived from stop density.
    """
    # SEPTA endpoints for stops/alerts - prefer using official SEPTA API when available
    SEPTA_ENDPOINT = endpoint or "https://hub.arcgis.com/api/v3/datasets/b227f3ddbe3e47b4bcc7b7c65ef2cef6_0/downloads/data?format=geojson&spatialRefId=4326&where=1%3D1"
    raw_path = raw_path or (FALLBACK_DIR / 'septa_stops.csv')

    df = None
    try:
        resp = requests.get(SEPTA_ENDPOINT, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        # If result contains records, create a DataFrame
        records = result.get('result', {}).get('records') if isinstance(result, dict) else None
        if records:
            df = pd.DataFrame.from_records(records)
    except Exception:
        if raw_path.exists():
            df = pd.read_csv(raw_path)

    if df is None:
        return gpd.GeoDataFrame(columns=['geometry'])

    lat_key = next((c for c in df.columns if c.lower() in ('lat', 'latitude', 'y', 'intptlat')), None)
    lon_key = next((c for c in df.columns if c.lower() in ('lon', 'longitude', 'x', 'intptlon')), None)
    df['lat'] = pd.to_numeric(df[lat_key], errors='coerce') if lat_key else pd.NA
    df['lon'] = pd.to_numeric(df[lon_key], errors='coerce') if lon_key else pd.NA
    df = df.dropna(subset=['lat', 'lon'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326')
    return gdf


# ---------------------------------------------------------------------------
# Food access (store points or USDA tract metrics)
# ---------------------------------------------------------------------------

def fetch_retail_food_stores(endpoint: Optional[str] = None, raw_path: Path | None = None) -> gpd.GeoDataFrame:
    """Fetch locations of grocery / supermarket stores in Philadelphia and return points.

    Attempts to use Philadelphia OpenData 'shopp' / 'grocerystore' dataset (placeholder)
    or fallback to a local CSV.
    """
    FOOD_ENDPOINT = endpoint or "https://data.phila.gov/resource/food_stores.json"
    raw_path = raw_path or (FALLBACK_DIR / 'food_stores.csv')

    df = None
    try:
        resp = requests.get(FOOD_ENDPOINT, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception:
        if raw_path.exists():
            df = pd.read_csv(raw_path)

    if df is None:
        return gpd.GeoDataFrame(columns=['geometry'])

    lat_key = next((c for c in df.columns if c.lower() in ('lat', 'latitude', 'y', 'intptlat')), None)
    lon_key = next((c for c in df.columns if c.lower() in ('lon', 'longitude', 'x', 'intptlon')), None)
    df['lat'] = pd.to_numeric(df[lat_key], errors='coerce') if lat_key else pd.NA
    df['lon'] = pd.to_numeric(df[lon_key], errors='coerce') if lon_key else pd.NA
    df = df.dropna(subset=['lat', 'lon'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326')
    return gdf


# ---------------------------------------------------------------------------
# HUD assisted properties
# ---------------------------------------------------------------------------

def fetch_hud_assisted_properties(endpoint: Optional[str] = None, raw_path: Path | None = None) -> gpd.GeoDataFrame:
    """Fetch HUD-assisted property points. Falls back to data/raw/hud_public_housing.csv

    The function returns points; the ingestion script will map to block groups and
    set `hud_assisted` = 1 for block groups that contain at least one HUD property.
    """
    HUD_ENDPOINT = endpoint or "https://www.huduser.gov/hudapi/public/chas"  # HUD CHAS/CHAS-like datasets (placeholder)
    raw_path = raw_path or (FALLBACK_DIR / 'hud_assisted.csv')

    df = None
    try:
        resp = requests.get(HUD_ENDPOINT, timeout=30)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
    except Exception:
        if raw_path.exists():
            df = pd.read_csv(raw_path)

    if df is None:
        return gpd.GeoDataFrame(columns=['geometry'])

    lat_key = next((c for c in df.columns if c.lower() in ('lat', 'latitude', 'y', 'intptlat')), None)
    lon_key = next((c for c in df.columns if c.lower() in ('lon', 'longitude', 'x', 'intptlon')), None)

    if lat_key:
        df['lat'] = pd.to_numeric(df[lat_key], errors='coerce')
    if lon_key:
        df['lon'] = pd.to_numeric(df[lon_key], errors='coerce')

    df = df.dropna(subset=['lat', 'lon'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326')
    return gdf


# ---------------------------------------------------------------------------
# Gini index via Census API
# ---------------------------------------------------------------------------

def fetch_block_group_gini(acs_year: int = 2023, state: str = '42', county: str = '101', api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch the ACS Gini index (B19083_001E) for block groups and return
    a dataframe with 'block_group_id' and 'gini_index'. Requires a Census API key
    via environment variable CENSUS_API_KEY if none provided.
    """
    if api_key is None:
        api_key = os.getenv('CENSUS_API_KEY')
    if not api_key:
        raise RuntimeError("Census API key not found. Set CENSUS_API_KEY in environment or pass --census-key")

    url = f"https://api.census.gov/data/{acs_year}/acs/acs5"
    params = {
        'get': 'NAME,B19083_001E',
        'for': 'block group:*',
        'in': f'state:{state}+county:{county}',
        'key': api_key,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    headers, *rows = payload
    df = pd.DataFrame(rows, columns=headers)
    df['block_group_id'] = df['state'] + df['county'] + df['tract'] + df['block group']
    df['gini_index'] = pd.to_numeric(df['B19083_001E'], errors='coerce')
    df = df[['block_group_id', 'gini_index']]
    return df


# ---------------------------------------------------------------------------
# Aggregate and write CSVs
# ---------------------------------------------------------------------------


def write_csv(df: pd.DataFrame, filename: str):
    path = DATA_EXTERNAL / filename
    df.to_csv(path, index=False)
    print(f"Wrote {path} with {len(df)} rows")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest optional external spatial layers for the CCA app")
    parser.add_argument('--no-crime', dest='crime', action='store_false', help='Skip crime fetch')
    parser.add_argument('--no-vacancy', dest='vacancy', action='store_false', help='Skip vacancy fetch')
    parser.add_argument('--no-transit', dest='transit', action='store_false', help='Skip transit fetch')
    parser.add_argument('--no-food', dest='food', action='store_false', help='Skip food fetch')
    parser.add_argument('--no-hud', dest='hud', action='store_false', help='Skip hud fetch')
    parser.add_argument('--no-gini', dest='gini', action='store_false', help='Skip gini fetch')
    parser.add_argument('--census-key', type=str, default=None, help='Census API key for gini fetch (optional)')
    # Optional endpoints - use these to supply authoritative sources rather than the placeholder dataset URLs
    parser.add_argument('--crime-endpoint', type=str, default=None, help='Custom crime dataset endpoint (Socrata/ArcGIS CSV/JSON)')
    parser.add_argument('--vacancy-endpoint', type=str, default=None, help='Custom vacant properties endpoint')
    parser.add_argument('--transit-endpoint', type=str, default=None, help='Custom transit stops endpoint')
    parser.add_argument('--food-endpoint', type=str, default=None, help='Custom food store endpoint')
    parser.add_argument('--hud-endpoint', type=str, default=None, help='Custom HUD assisted property endpoint')
    parser.add_argument('--raw-dir', type=str, default=str(FALLBACK_DIR), help='Directory for local raw fallback files')
    # crime SQL date range and format
    parser.add_argument('--crime-start', dest='crime_start', type=str, default=None, help='SQL start date for crime fetch (e.g., 2025-10-01 00:00:00)')
    parser.add_argument('--crime-end', dest='crime_end', type=str, default=None, help='SQL end date for crime fetch (e.g., 2025-11-01 00:00:00)')
    parser.add_argument('--crime-format', dest='crime_format', type=str, default='geojson', help='Format for Carto query (geojson|csv)')

    args = parser.parse_args(argv)

    bg_gdf = load_block_group_gdf()
    # Allow user override for fallback raw dir
    raw_dir = Path(args.raw_dir) if args.raw_dir else FALLBACK_DIR

    # ----------------------------------------------------------------------
    # Crime
    # ----------------------------------------------------------------------
    if args.crime:
        try:
            print("Fetching crime incidents... (Carto SQL)")
            crime_gdf = fetch_philly_crime_data(start=args.crime_start, end=args.crime_end, fmt=args.crime_format, endpoint=args.crime_endpoint, raw_path=Path(raw_dir) / 'crime_incidents.csv')
            crime_agg = aggregate_points_to_bg(crime_gdf, bg_gdf, value_col=None, agg_name='crimes')
            if 'crimes' not in crime_agg.columns:
                crime_agg['crimes'] = 0
            # If available, convert to crimes per 1k population using demographics
            demo_path = ROOT / 'demographics_block_groups.csv'
            if demo_path.exists():
                demo = pd.read_csv(demo_path)
                if 'block_group_id' in demo.columns and 'total_pop' in demo.columns:
                    pop_map = demo.set_index('block_group_id')['total_pop'].to_dict()
                    crime_agg['total_pop'] = crime_agg['block_group_id'].map(pop_map).fillna(0)
                    # crimes per 1k residents (guard against divide-by-zero)
                    crime_agg['crimes_per_1k'] = crime_agg.apply(
                        lambda r: (r['crimes'] / r['total_pop']) * 1000 if r['total_pop'] > 0 else 0,
                        axis=1
                    )
                else:
                    # No population available – normalize counts per thousand using unit factor
                    crime_agg['crimes_per_1k'] = crime_agg['crimes']
            else:
                crime_agg['crimes_per_1k'] = crime_agg['crimes']
            write_csv(crime_agg[['block_group_id', 'crimes_per_1k']], 'crime_per_block_group.csv')
        except Exception as exc:  # pragma: no cover - network & service availability dependent
            print(f"[crime] Error: {exc}")
            # Fallback: write empty crimes_per_1k file so app can load gracefully
            write_csv(pd.DataFrame(columns=['block_group_id', 'crimes_per_1k']), 'crime_per_block_group.csv')

    # ----------------------------------------------------------------------
    # Vacancy
    # ----------------------------------------------------------------------
    if args.vacancy:
        try:
            print("Fetching vacancy points...")
            vac_gdf = fetch_vacant_properties(endpoint=args.vacancy_endpoint, raw_path=Path(raw_dir) / 'vacant_properties.csv')
            vac_agg = aggregate_points_to_bg(vac_gdf, bg_gdf, value_col='vacant', agg_name='vacant_count')
            # Compute vacancy percent if housing unit info exists in demographics or 0 otherwise
            # Try to load demographics to get housing units if present
            demo_path = ROOT / 'demographics_block_groups.csv'
            if demo_path.exists():
                demo = pd.read_csv(demo_path)
                if 'block_group_id' in demo.columns:
                    housing_units = demo.set_index('block_group_id').get('total_housing_units')
                else:
                    housing_units = pd.Series()
            else:
                housing_units = pd.Series()

            vac_agg['vacant_count'] = pd.to_numeric(vac_agg['vacant_count'], errors='coerce').fillna(0).astype(int)
            if not housing_units.empty:
                vac_agg['vacant_pct'] = vac_agg['block_group_id'].map(housing_units).fillna(0)
                vac_agg['vacant_pct'] = (vac_agg['vacant_count'] / vac_agg['vacant_pct']).fillna(0) * 100.0
            else:
                vac_agg['vacant_pct'] = (vac_agg['vacant_count'] / 1.0).fillna(0)
            write_csv(vac_agg[['block_group_id', 'vacant_count', 'vacant_pct']], 'vacancy_block_groups.csv')
        except Exception as exc:
            print(f"[vacancy] Error: {exc}")
            write_csv(pd.DataFrame(columns=['block_group_id', 'vacant_count', 'vacant_pct']), 'vacancy_block_groups.csv')

    # ----------------------------------------------------------------------
    # Transit
    # ----------------------------------------------------------------------
    if args.transit:
        try:
            print("Fetching transit stops...")
            transit_gdf = fetch_transit_stops(endpoint=args.transit_endpoint, raw_path=Path(raw_dir) / 'septa_stops.csv')
            transit_agg = aggregate_points_to_bg(transit_gdf, bg_gdf, value_col=None, agg_name='transit_stop_count')
            # Normalize using MinMax across blocks to create transit_access_score
            transit_agg['transit_stop_count'] = pd.to_numeric(transit_agg.get('transit_stop_count', 0), errors='coerce').fillna(0)
            if not transit_agg.empty:
                mm = (transit_agg['transit_stop_count'] - transit_agg['transit_stop_count'].min()) / (
                    transit_agg['transit_stop_count'].max() - transit_agg['transit_stop_count'].min() if transit_agg['transit_stop_count'].max() != transit_agg['transit_stop_count'].min() else 1
                )
                transit_agg['transit_access_score'] = mm.fillna(0.0)
            else:
                transit_agg['transit_access_score'] = 0.0

            write_csv(transit_agg[['block_group_id', 'transit_stop_count', 'transit_access_score']], 'transit_access_block_groups.csv')
        except Exception as exc:
            print(f"[transit] Error: {exc}")
            write_csv(pd.DataFrame(columns=['block_group_id', 'transit_stop_count', 'transit_access_score']), 'transit_access_block_groups.csv')

    # ----------------------------------------------------------------------
    # Food access
    # ----------------------------------------------------------------------
    if args.food:
        try:
            print("Fetching food stores...")
            food_gdf = fetch_retail_food_stores(endpoint=args.food_endpoint, raw_path=Path(raw_dir) / 'food_stores.csv')
            food_agg = aggregate_points_to_bg(food_gdf, bg_gdf, value_col=None, agg_name='grocery_count')
            # Normalize as food_access_score
            food_agg['grocery_count'] = pd.to_numeric(food_agg.get('grocery_count', 0), errors='coerce').fillna(0)
            if not food_agg.empty:
                mm = (food_agg['grocery_count'] - food_agg['grocery_count'].min()) / (
                    food_agg['grocery_count'].max() - food_agg['grocery_count'].min() if food_agg['grocery_count'].max() != food_agg['grocery_count'].min() else 1
                )
                food_agg['food_access_score'] = mm.fillna(0.0)
            else:
                food_agg['food_access_score'] = 0.0

            write_csv(food_agg[['block_group_id', 'grocery_count', 'food_access_score']], 'food_access_block_groups.csv')
        except Exception as exc:
            print(f"[food] Error: {exc}")
            write_csv(pd.DataFrame(columns=['block_group_id', 'grocery_count', 'food_access_score']), 'food_access_block_groups.csv')

    # ----------------------------------------------------------------------
    # Gini using Census API (B19083)
    # ----------------------------------------------------------------------
    if args.gini:
        try:
            print("Fetching Gini index from Census (B19083)...")
            # For Philadelphia County (101) - you can change county if needed via CLI
            gini_df = fetch_block_group_gini(api_key=args.census_key)
            write_csv(gini_df[['block_group_id', 'gini_index']], 'gini_block_groups.csv')
        except Exception as exc:
            print(f"[gini] Error: {exc}")
            write_csv(pd.DataFrame(columns=['block_group_id', 'gini_index']), 'gini_block_groups.csv')

    # ----------------------------------------------------------------------
    # HUD assisted properties
    # ----------------------------------------------------------------------
    if args.hud:
        try:
            print("Fetching HUD assisted housing points...")
            hud_gdf = fetch_hud_assisted_properties(endpoint=args.hud_endpoint, raw_path=Path(raw_dir) / 'hud_assisted.csv')
            hud_agg = aggregate_points_to_bg(hud_gdf, bg_gdf, value_col=None, agg_name='hud_count')
            # Create a HUD presence flag
            hud_agg['hud_assisted'] = (hud_agg.get('hud_count', 0) > 0).astype(int)
            write_csv(hud_agg[['block_group_id', 'hud_assisted']], 'hud_assisted_block_groups.csv')
        except Exception as exc:
            print(f"[hud] Error: {exc}")
            write_csv(pd.DataFrame(columns=['block_group_id', 'hud_assisted']), 'hud_assisted_block_groups.csv')

    print("External layer ingestion complete.")
    return 0


if __name__ == '__main__':  # pragma: no cover - CLI entry
    raise SystemExit(main())
