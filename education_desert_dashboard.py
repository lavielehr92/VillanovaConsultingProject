from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import shapefile
import streamlit as st


# -----------------------------
# Constants and simple helpers
# -----------------------------

AVAILABLE_YEARS: Sequence[int] = tuple(range(2014, 2024))
PHILADELPHIA_STATE_FIPS = "42"
PHILADELPHIA_COUNTY_FIPS = "101"
PHILADELPHIA_COUNTY_NAME = "Philadelphia County"
PHILADELPHIA_TRACTS_ZIP_URL = "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_42_tract.zip"
PHILADELPHIA_TRACT_FILE_PREFIX = "tl_2023_42_tract"
STUDENT_ADDRESS_PATH = Path("Cornerstone/CCA addresses.csv")
GEOCODER_ENDPOINT = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
SCHOOL_LOCATIONS: Sequence[Tuple[str, float, float]] = (
    ("Cornerstone Christian Academy (58th St)", 39.9350198, -75.2272841),
    ("Cornerstone Christian Academy (Baltimore Ave)", 39.9499049, -75.2060104),
)


# -----------------------------
# Variable specs
# -----------------------------

@dataclass(frozen=True)
class VariableSpec:
    alias: str
    dataset: str           # "acs/acs5" or "acs/acs5/subject"
    group: str             # e.g., "S1501"
    search_terms: Sequence[str] = ()
    explicit_id: Optional[str] = None


VARIABLE_SPECS: Sequence[VariableSpec] = (
    # Subject table: Educational attainment (percent)
    VariableSpec(
        alias="pct_no_high_school",
        dataset="acs/acs5/subject",
        group="S1501",
        search_terms=("Percent", "Less than high school graduate"),
    ),
    VariableSpec(
        alias="pct_bachelors_or_higher",
        dataset="acs/acs5/subject",
        group="S1501",
        search_terms=("Percent", "Bachelor's degree or higher"),
    ),
    # Vehicles (detailed table)
    VariableSpec(
        alias="households_total",
        dataset="acs/acs5",
        group="B08201",
        explicit_id="B08201_001E",
    ),
    VariableSpec(
        alias="households_no_vehicle",
        dataset="acs/acs5",
        group="B08201",
        explicit_id="B08201_002E",
    ),
    # Internet subscription (subject table)
    VariableSpec(
        alias="pct_no_internet",
        dataset="acs/acs5/subject",
        group="S2801",
        explicit_id="S2801_C02_019E",
    ),
    # Income, population, children
    VariableSpec(
        alias="median_household_income",
        dataset="acs/acs5",
        group="B19013",
        explicit_id="B19013_001E",
    ),
    VariableSpec(
        alias="population_under_18",
        dataset="acs/acs5",
        group="B09001",
        explicit_id="B09001_001E",
    ),
    VariableSpec(
        alias="total_population",
        dataset="acs/acs5",
        group="B01003",
        explicit_id="B01003_001E",
    ),
)

# Fallback map for tricky subject table variables
VARIABLE_FALLBACK_IDS: Dict[str, str] = {
    # S2801 percent, "Without an Internet subscription"
    "pct_no_internet": "S2801_C02_019E",
}


# -----------------------------
# Census API utilities
# -----------------------------

@lru_cache(maxsize=None)
def fetch_group_metadata(year: int, dataset: str, group: str) -> Dict[str, Dict[str, str]]:
    if dataset == "acs/acs5/subject":
        url = f"https://api.census.gov/data/{year}/acs/acs5/subject/groups/{group}.json"
    else:
        url = f"https://api.census.gov/data/{year}/acs/acs5/groups/{group}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return payload["variables"]


def resolve_variable_id(year: int, spec: VariableSpec) -> str:
    """Find a variable id by label text, or fall back to a pinned id."""
    if spec.explicit_id:
        return spec.explicit_id

    metadata = fetch_group_metadata(year, spec.dataset, spec.group)
    candidates: List[str] = []
    for var_id, meta in metadata.items():
        label = meta.get("label", "")
        # Simple contains match for all provided terms
        if all(term.lower() in label.lower() for term in spec.search_terms):
            candidates.append(var_id)

    # prefer estimates (end with 'E')
    candidates = sorted(v for v in candidates if v.endswith("E"))

    if not candidates:
        fallback = VARIABLE_FALLBACK_IDS.get(spec.alias)
        if fallback:
            return fallback
        raise ValueError(f"No variable found for {spec.alias} using terms {spec.search_terms}")

    return candidates[0]


def _build_geo_keys(geography: str) -> List[str]:
    if geography == "county":
        return ["state", "county"]
    if geography == "tract":
        return ["state", "county", "tract"]
    raise NotImplementedError("Unsupported geography level.")


def call_census_api(
    year: int,
    dataset: str,
    variable_ids: Sequence[str],
    state_fips: str,
    county_fips: Optional[str] = None,
    geography: str = "county",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    if dataset == "acs/acs5/subject":
        base_url = f"{base_url}/subject"

    params: Dict[str, Iterable[str] | str] = {
        "get": ",".join(["NAME", *variable_ids]),
    }
    if geography == "county":
        params["for"] = "county:*"
        params["in"] = f"state:{state_fips}"  # must be a string, not a list
    elif geography == "tract":
        if not county_fips:
            raise ValueError("county_fips is required when geography='tract'")
        params["for"] = "tract:*"
        params["in"] = f"state:{state_fips} county:{county_fips}"
    else:
        raise NotImplementedError("Unsupported geography level.")
    if api_key:
        params["key"] = api_key

    response = requests.get(base_url, params=params, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Census API error {response.status_code}: {response.text}")

    payload = response.json()
    headers = payload[0]
    rows = payload[1:]
    frame = pd.DataFrame(rows, columns=headers)
    return frame


def fetch_acs_bundle(
    year: int,
    state_fips: str,
    specs: Sequence[VariableSpec],
    county_fips: Optional[str],
    geography: str,
    api_key: Optional[str],
) -> pd.DataFrame:
    geo_keys = _build_geo_keys(geography)

    # Group specs by dataset to minimize API calls
    grouped: Dict[str, List[VariableSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.dataset, []).append(spec)

    merged: Optional[pd.DataFrame] = None
    resolved_ids: Dict[str, str] = {}

    for dataset, dataset_specs in grouped.items():
        # Resolve all variables for this dataset
        ids = []
        for spec in dataset_specs:
            var_id = resolve_variable_id(year, spec)
            resolved_ids[spec.alias] = var_id
            ids.append(var_id)

        # Pull and rename
        raw = call_census_api(
            year,
            dataset,
            ids,
            state_fips,
            county_fips=county_fips,
            geography=geography,
            api_key=api_key,
        )
        rename_map = {resolved_ids[s.alias]: s.alias for s in dataset_specs}
        raw = raw.rename(columns=rename_map)

        # Make numeric
        for col in rename_map.values():
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

        if merged is None:
            # Keep NAME only once
            merged = raw
        else:
            right = raw.drop(columns=["NAME"])
            # Do not drop join keys
            missing = [k for k in geo_keys if k not in right.columns]
            if missing:
                raise KeyError(f"Missing join keys {missing} in right-hand frame")
            merged = merged.merge(right, on=geo_keys, how="left")

    assert merged is not None

    # Standard FIPS formats
    merged["county"] = merged["county"].astype(str).str.zfill(3)
    merged["state"] = merged["state"].astype(str).str.zfill(2)
    if "tract" in merged.columns:
        merged["tract"] = merged["tract"].astype(str).str.zfill(6)
        merged["geoid"] = merged["state"] + merged["county"] + merged["tract"]
    else:
        merged["geoid"] = merged["state"] + merged["county"]
    return merged


# -----------------------------
# Metric calculations
# -----------------------------

def z_score(series: pd.Series, invert: bool = False) -> pd.Series:
    values = series.astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        scores = pd.Series(0.0, index=values.index)
    else:
        scores = (values - mean) / std
    if invert:
        scores = scores * -1
    return scores


def min_max_scale(series: pd.Series, target_min: float, target_max: float) -> pd.Series:
    values = series.astype(float)
    min_val = values.min()
    max_val = values.max()
    if np.isnan(min_val) or np.isnan(max_val) or min_val == max_val:
        midpoint = (target_max - target_min) / 2.0 + target_min
        return pd.Series(midpoint, index=values.index)
    scaled = (values - min_val) / (max_val - min_val)
    return scaled * (target_max - target_min) + target_min


def assign_quantile_labels(series: pd.Series, labels: Sequence[str]) -> pd.Series:
    try:
        return pd.qcut(series, q=len(labels), labels=labels)
    except ValueError:
        return pd.Series(labels[-1], index=series.index)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    # Derived percents
    frame["pct_no_vehicle"] = np.where(
        frame["households_total"] > 0,
        (frame["households_no_vehicle"] / frame["households_total"]) * 100,
        np.nan,
    )
    frame["pct_children"] = np.where(
        frame["total_population"] > 0,
        (frame["population_under_18"] / frame["total_population"]) * 100,
        np.nan,
    )

    # Ensure numeric
    metric_columns = [
        "pct_no_high_school",
        "pct_bachelors_or_higher",
        "pct_no_vehicle",
        "pct_no_internet",
        "pct_children",
        "median_household_income",
    ]
    for col in metric_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Scale income in thousands for display
    frame["median_household_income"] = frame["median_household_income"] / 1000.0

    # Scores
    frame["need_score"] = z_score(frame["pct_no_high_school"]) + z_score(frame["pct_children"])
    frame["choice_gap_score"] = z_score(frame["pct_no_high_school"]) - z_score(frame["pct_bachelors_or_higher"])
    frame["access_friction_score"] = (
        z_score(frame["pct_no_vehicle"])
        + z_score(frame["pct_no_internet"])
        + z_score(frame["median_household_income"], invert=True)
    )

    # Composite and labels
    frame["education_desert_index_raw"] = (
        frame["need_score"] + frame["choice_gap_score"] + frame["access_friction_score"]
    ) / 3.0
    frame["education_desert_index_scaled"] = min_max_scale(
        frame["education_desert_index_raw"], target_min=0, target_max=100
    )
    frame["education_desert_label"] = assign_quantile_labels(
        frame["education_desert_index_scaled"], labels=("Lower", "Moderate", "Higher")
    )
    return frame


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    try:
        values = [float(lat1), float(lon1), float(lat2), float(lon2)]
    except (TypeError, ValueError):
        return float("nan")
    if any(np.isnan(val) for val in values):
        return float("nan")
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, values)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 3958.7613 * c


def nearest_campus_distance(lat: float, lon: float) -> float:
    distances = [
        haversine_miles(lat, lon, campus_lat, campus_lon)
        for _, campus_lat, campus_lon in SCHOOL_LOCATIONS
    ]
    distances = [dist for dist in distances if not np.isnan(dist)]
    if not distances:
        return float("nan")
    return min(distances)


def _read_student_addresses() -> List[str]:
    if not STUDENT_ADDRESS_PATH.exists():
        return []
    series = pd.read_csv(STUDENT_ADDRESS_PATH, header=None, names=["address"], dtype=str)["address"]
    addresses = sorted({addr.strip() for addr in series.dropna() if addr.strip()})
    return addresses


def _geocode_single_address(address: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    query = address
    if "PA" not in address.upper():
        query = f"{address}, Philadelphia, PA"
    params = {
        "address": query,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_ACS",
        "format": "json",
    }
    try:
        response = requests.get(GEOCODER_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None, None, None
    payload = response.json()
    matches = payload.get("result", {}).get("addressMatches", [])
    if not matches:
        return None, None, None
    match = matches[0]
    coords = match.get("coordinates", {})
    try:
        lat = float(coords.get("y"))
    except (TypeError, ValueError):
        lat = float("nan")
    try:
        lon = float(coords.get("x"))
    except (TypeError, ValueError):
        lon = float("nan")
    tract_geoid: Optional[str] = None
    geographies = match.get("geographies", {})
    for key in (
        "Census Tracts",
        "Census Tracts 2020",
        "Census Blocks 2020",
        "Census Blocks",
    ):
        entries = geographies.get(key)
        if entries:
            tract_geoid = entries[0].get("GEOID")
            if tract_geoid and len(tract_geoid) > 11:
                tract_geoid = tract_geoid[:11]
            break
    return lat, lon, tract_geoid


@st.cache_data(show_spinner=True, ttl=86400)
def cached_student_locations() -> pd.DataFrame:
    addresses = _read_student_addresses()
    if not addresses:
        return pd.DataFrame(columns=["address", "latitude", "longitude", "tract_geoid"])
    records: List[Dict[str, object]] = []
    for address in addresses:
        lat, lon, tract_geoid = _geocode_single_address(address)
        records.append(
            {
                "address": address,
                "latitude": lat,
                "longitude": lon,
                "tract_geoid": tract_geoid,
            }
        )
    frame = pd.DataFrame(records)
    frame["tract_geoid"] = frame["tract_geoid"].astype(str).str.strip()
    frame.loc[frame["tract_geoid"] == "nan", "tract_geoid"] = np.nan
    return frame


# -----------------------------
# Caching wrappers
# -----------------------------

def load_philadelphia_geojson() -> Tuple[Dict[str, object], Dict[str, Tuple[float, float]]]:
    response = requests.get(PHILADELPHIA_TRACTS_ZIP_URL, timeout=120)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        def _read_member(extension: str) -> BytesIO:
            for name in archive.namelist():
                if name.endswith(f".{extension}") and PHILADELPHIA_TRACT_FILE_PREFIX in name:
                    return BytesIO(archive.read(name))
            raise FileNotFoundError(f"Missing .{extension} in tract shapefile archive")

        shp_bytes = _read_member("shp")
        shx_bytes = _read_member("shx")
        dbf_bytes = _read_member("dbf")

    reader = shapefile.Reader(shp=shp_bytes, shx=shx_bytes, dbf=dbf_bytes)
    features: List[Dict[str, object]] = []
    centroid_lookup: Dict[str, Tuple[float, float]] = {}

    for shape_record in reader.shapeRecords():
        record = shape_record.record.as_dict()
        if record.get("COUNTYFP") != PHILADELPHIA_COUNTY_FIPS:
            continue
        geoid = record.get("GEOID")
        geometry = shape_record.shape.__geo_interface__
        props = {
            "GEOID": geoid,
            "NAME": record.get("NAME"),
            "COUNTYFP": record.get("COUNTYFP"),
            "INTPTLAT": record.get("INTPTLAT"),
            "INTPTLON": record.get("INTPTLON"),
        }
        features.append({"type": "Feature", "geometry": geometry, "properties": props})
        try:
            lat = float(record.get("INTPTLAT"))
            lon = float(record.get("INTPTLON"))
        except (TypeError, ValueError):
            lat = float("nan")
            lon = float("nan")
        if geoid:
            centroid_lookup[geoid] = (lat, lon)

    return {"type": "FeatureCollection", "features": features}, centroid_lookup


@st.cache_data(show_spinner=False)
def cached_geojson() -> Tuple[Dict[str, object], Dict[str, Tuple[float, float]]]:
    return load_philadelphia_geojson()


@st.cache_data(show_spinner=True, ttl=86400)
def cached_education_dataframe(year: int, api_key: Optional[str]) -> pd.DataFrame:
    raw = fetch_acs_bundle(
        year,
        PHILADELPHIA_STATE_FIPS,
        VARIABLE_SPECS,
        county_fips=PHILADELPHIA_COUNTY_FIPS,
        geography="tract",
        api_key=api_key,
    )
    enriched = compute_metrics(raw)
    return enriched


# -----------------------------
# UI helpers
# -----------------------------

def render_summary_cards(df: pd.DataFrame) -> None:
    st.subheader("Highest Education Desert Index")
    if df.empty:
        st.info("No data available for the selected filters.")
        return
    top = df.sort_values("education_desert_index_scaled", ascending=False).head(3)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top.iterrows()):
        student_count = int(row["student_count"]) if pd.notna(row["student_count"]) else 0
        distance = row["dist_to_nearest_campus"]
        distance_str = f"{distance:.1f} mi" if pd.notna(distance) else "n/a"
        cols[idx].metric(
            label=row["NAME"],
            value=f"{row['education_desert_index_scaled']:.1f}",
            delta=f"Students {student_count} | {distance_str}",
        )


def render_map(
    df: pd.DataFrame,
    geojson: Dict[str, object],
    show_students: bool,
    student_points: pd.DataFrame,
) -> None:
    if df.empty:
        st.warning("Map cannot be drawn because no tracts match the current filters.")
        return

    hover_columns = [
        "NAME",
        "education_desert_index_scaled",
        "education_desert_label",
        "pct_no_high_school",
        "pct_bachelors_or_higher",
        "pct_no_vehicle",
        "pct_no_internet",
        "pct_children",
        "median_household_income",
        "student_count",
        "dist_to_nearest_campus",
    ]
    custom_data = df[hover_columns].to_numpy()

    choropleth = go.Choropleth(
        geojson=geojson,
        featureidkey="properties.GEOID",
        locations=df["geoid"],
        z=df["education_desert_index_scaled"],
        colorscale="YlOrRd",
        marker_line_color="#666666",
        marker_line_width=0.2,
        colorbar_title="Education Desert Index",
        customdata=custom_data,
        hovertemplate=(
            "%{customdata[0]}<br>Index %{z:.1f} (%{customdata[2]})"
            "<br>% Adults < HS %{customdata[3]:.1f}"
            "<br>% Bachelor's+ %{customdata[4]:.1f}"
            "<br>% HHs No Vehicle %{customdata[5]:.1f}"
            "<br>% HHs No Internet %{customdata[6]:.1f}"
            "<br>% Population < 18 %{customdata[7]:.1f}"
            "<br>Median Income ($000) %{customdata[8]:.1f}"
            "<br>Current Students %{customdata[9]}"
            "<br>Nearest Campus (mi) %{customdata[10]:.2f}<extra></extra>"
        ),
    )

    fig = go.Figure(data=[choropleth])
    fig.update_geos(fitbounds="locations", visible=False)

    campus_lats = [item[1] for item in SCHOOL_LOCATIONS]
    campus_lons = [item[2] for item in SCHOOL_LOCATIONS]
    campus_labels = [item[0] for item in SCHOOL_LOCATIONS]
    fig.add_trace(
        go.Scattergeo(
            lat=campus_lats,
            lon=campus_lons,
            text=campus_labels,
            mode="markers",
            marker=dict(size=14, color="#f1c40f", symbol="star"),
            name="Cornerstone campuses",
            hovertemplate="%{text}<extra></extra>",
        )
    )

    if show_students and not student_points.empty:
        fig.add_trace(
            go.Scattergeo(
                lat=student_points["latitude"],
                lon=student_points["longitude"],
                text=student_points["address"],
                mode="markers",
                marker=dict(size=5, color="#1f77b4", opacity=0.6),
                name="Current students",
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.0),
        height=650,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_data_table(df: pd.DataFrame) -> None:
    columns = [
        "NAME",
        "education_desert_index_scaled",
        "education_desert_label",
        "student_count",
        "dist_to_nearest_campus",
        "pct_no_high_school",
        "pct_bachelors_or_higher",
        "pct_no_vehicle",
        "pct_no_internet",
        "pct_children",
        "median_household_income",
    ]
    display_df = df[columns].rename(
        columns={
            "NAME": "Census tract",
            "education_desert_index_scaled": "Education Desert Index",
            "education_desert_label": "Segment",
            "student_count": "Current students",
            "dist_to_nearest_campus": "Nearest campus (mi)",
            "pct_no_high_school": "% Adults < HS",
            "pct_bachelors_or_higher": "% Bachelor's+",
            "pct_no_vehicle": "% HHs No Vehicle",
            "pct_no_internet": "% HHs No Internet",
            "pct_children": "% Population < 18",
            "median_household_income": "Median Income ($000)",
        }
    )
    st.dataframe(display_df, use_container_width=True)


def render_download_button(df: pd.DataFrame) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download data as CSV",
        data=csv_bytes,
        file_name="education_desert_metrics.csv",
        mime="text/csv",
    )


# -----------------------------
# App
# -----------------------------

def main() -> None:
    st.set_page_config(page_title="Education Desert Dashboard", layout="wide")
    st.title("Education Desert Dashboard")
    st.caption("Philadelphia census tracts; indexes are relative within the city for the selected ACS vintage.")

    geojson, centroid_lookup = cached_geojson()
    students_df = cached_student_locations()
    if not students_df.empty:
        tract_counts = students_df.dropna(subset=["tract_geoid"])["tract_geoid"].value_counts()
        max_student_count = int(tract_counts.max()) if not tract_counts.empty else 0
    else:
        max_student_count = 0

    with st.sidebar:
        st.header("Filters")
        year = st.selectbox("ACS 5-year vintage", options=AVAILABLE_YEARS, index=len(AVAILABLE_YEARS) - 1)
        api_key_default = os.getenv("CENSUS_API_KEY") or os.getenv("CensusBureauAPI_KEY", "")
        api_key_input = st.text_input("Census API key (optional)", value=api_key_default, type="password")
        min_index = st.slider("Minimum Education Desert Index", 0, 100, 50, step=1)
        max_income = st.slider("Maximum median income ($000)", 20, 150, 90, step=5)
        min_internet = st.slider("Minimum % of households without internet", 0, 60, 15, step=1)
        min_vehicle = st.slider("Minimum % of households without a vehicle", 0, 60, 10, step=1)
        segments = ("Higher", "Moderate", "Lower")
        selected_segments = st.multiselect("Segments to include", segments, default=list(segments))
        max_distance = st.slider("Max distance to a Cornerstone campus (miles)", 0.5, 15.0, 10.0, step=0.5)
        min_students = st.slider("Minimum current student count", 0, max(5, max_student_count), 0, step=1)
        require_students = st.checkbox("Only show tracts with existing students", False)
        show_students = st.checkbox("Show current student locations on the map", True)
        st.caption(f"Student addresses geocoded: {len(students_df)}")

    api_key = api_key_input.strip() if api_key_input else None
    df = cached_education_dataframe(year, api_key if api_key else None)

    df["centroid_lat"] = df["geoid"].map(lambda g: centroid_lookup.get(g, (float("nan"), float("nan")))[0])
    df["centroid_lon"] = df["geoid"].map(lambda g: centroid_lookup.get(g, (float("nan"), float("nan")))[1])

    student_counts = (
        students_df.dropna(subset=["tract_geoid"])["tract_geoid"].value_counts().astype(int)
        if not students_df.empty
        else pd.Series(dtype=int)
    )
    df["student_count"] = df["geoid"].map(student_counts).fillna(0).astype(int)
    df["dist_to_nearest_campus"] = df.apply(
        lambda row: nearest_campus_distance(row["centroid_lat"], row["centroid_lon"]), axis=1
    )

    view_df = df.copy()
    view_df = view_df[view_df["education_desert_index_scaled"] >= min_index]
    view_df = view_df[view_df["median_household_income"] <= max_income]
    view_df = view_df[view_df["pct_no_internet"] >= min_internet]
    view_df = view_df[view_df["pct_no_vehicle"] >= min_vehicle]
    if selected_segments:
        view_df = view_df[view_df["education_desert_label"].isin(selected_segments)]
    else:
        view_df = view_df.iloc[0:0]
    if max_distance < 15.0:
        view_df = view_df[
            view_df["dist_to_nearest_campus"].isna()
            | (view_df["dist_to_nearest_campus"] <= max_distance)
        ]
    if min_students > 0:
        view_df = view_df[view_df["student_count"] >= min_students]
    if require_students:
        view_df = view_df[view_df["student_count"] > 0]

    filtered_student_points = students_df.copy()
    if not view_df.empty:
        filtered_student_points = filtered_student_points[filtered_student_points["tract_geoid"].isin(view_df["geoid"])]
    filtered_student_points = filtered_student_points.dropna(subset=["latitude", "longitude"])

    render_summary_cards(view_df)
    render_map(view_df, geojson, show_students, filtered_student_points)
    render_data_table(view_df)
    render_download_button(view_df)

    st.markdown(
        """
        **Methodology**     - Need Score combines percent of adults without a high school diploma and share of residents under 18.
        - Choice Gap contrasts low educational attainment against bachelor's+ completion.
        - Access Friction blends vehicle access, internet subscription, and median income (inverted).
        - Education Desert Index is a scaled composite of the three scores (city-relative).
        Data sources: ACS 5-year Subject Tables S1501, S2801 and Detailed Tables B08201, B09001, B01003, B19013. Student addresses geocoded via the U.S. Census Geocoding API.
        """
    )


if __name__ == "__main__":
    main()