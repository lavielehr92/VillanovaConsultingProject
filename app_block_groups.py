"""
Philadelphia Educational Desert Explorer - Block Group Version
CCA Expansion Analysis Dashboard with Choropleth Visualization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import geopandas as gpd
import json
import os
from pathlib import Path
import requests
from typing import Tuple
from educational_desert_index_bg import compute_edi_block_groups, haversine_km
import numpy as np
from sklearn.preprocessing import MinMaxScaler
try:
    from competition_ingest import load_competition_schools
except ModuleNotFoundError:
    def load_competition_schools(*, output_path: str | None = None, **_: object) -> pd.DataFrame:
        """Fallback loader when competition_ingest isn't available.

        Streamlit Cloud runners only include tracked repo files, so if the optional
        ingestion utility isn't deployed we fall back to the cached CSV that ships
        with the app.  Returning an empty frame keeps the map working even when the
        file is missing.
        """

        csv_path = output_path or "competition_schools.csv"
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path)
            except Exception:
                pass
        return pd.DataFrame(columns=[
            "school_name", "type", "grades", "address", "notable_info",
            "capacity_hint", "lat", "lon", "capacity"
        ])

try:
    from school_ingest import load_census_schools
except ModuleNotFoundError:
    def load_census_schools(*, output_path: str | None = None, **_: object) -> pd.DataFrame:
        csv_path = output_path or "census_schools.csv"
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path)
            except Exception:
                pass
        return pd.DataFrame(columns=["school_name", "type", "lat", "lon", "capacity"])
try:
    # Allow optional live refresh import
    from fetch_block_groups_live import main as fetch_live_block_groups
except Exception:
    fetch_live_block_groups = None

STATE_FIPS = "42"  # Pennsylvania
COUNTY_FIPS = "101"  # Philadelphia County
ACS_YEAR = "2023"
ACS_ACS5_ENDPOINT = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
ACS_SUBJECT_ENDPOINT = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5/subject"

BG_AGE_FIELDS = {
    "B01001_004E": "male_5_9",
    "B01001_005E": "male_10_14",
    "B01001_006E": "male_15_17",
    "B01001_028E": "female_5_9",
    "B01001_029E": "female_10_14",
    "B01001_030E": "female_15_17",
}

TRACT_ENROLLMENT_FIELDS = {
    # per ACS S1401 documentation: kindergarten, grades 1-8, grades 9-12 enrollment counts
    "S1401_C01_004E": "kindergarten_total",
    "S1401_C01_005E": "grades_1_to_8_total",
    "S1401_C01_006E": "grades_9_to_12_total",
}

DATA_CACHE_DIR = Path("data/cache")
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Helper function to get Census API key from multiple sources
def get_census_api_key():
    """Try to get Census API key from Streamlit secrets, env vars, or .env file"""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and 'CENSUS_API_KEY' in st.secrets:
            return st.secrets['CENSUS_API_KEY']
    except Exception:
        pass
    
    # Try environment variables
    for name in ["CENSUS_API_KEY", "CensusBureauAPI_KEY", "CENSUSBUREAUAPI_KEY"]:
        key = os.getenv(name)
        if key:
            return key
    
    # Try .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        load_dotenv(dotenv_path="MyKeys/.env")
        for name in ["CENSUS_API_KEY", "CensusBureauAPI_KEY"]:
            key = os.getenv(name)
            if key:
                return key
    except:
        pass
    
    return None


def _acs_request(url: str, params: dict) -> list:
    api_key = get_census_api_key()
    if api_key:
        params = {**params, "key": api_key}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=86400)
def fetch_block_group_age_data() -> Tuple[pd.DataFrame, dict]:
    """Retrieve ACS block-group counts for population age 5-17."""

    params = {
        "get": ",".join(list(BG_AGE_FIELDS.keys()) + ["NAME"]),
        "for": "block group:*",
        "in": f"state:{STATE_FIPS}+county:{COUNTY_FIPS}+tract:*",
    }
    payload = _acs_request(ACS_ACS5_ENDPOINT, params)
    headers = payload[0]
    expected = list(BG_AGE_FIELDS.keys()) + ["NAME", "state", "county", "tract", "block group"]
    missing = [col for col in expected if col not in headers]
    if missing:
        raise ValueError(f"Missing required ACS variables for block group ages: {', '.join(missing)}")

    df = pd.DataFrame(payload[1:], columns=headers)
    df["block_group_id"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    df["tract_id"] = df["state"] + df["county"] + df["tract"]

    for code, alias in BG_AGE_FIELDS.items():
        df[alias] = pd.to_numeric(df[code], errors="coerce")

    df["bg_age_5_17"] = df[list(BG_AGE_FIELDS.values())].fillna(0).sum(axis=1)
    df["is_modeled"] = True

    timestamp = pd.Timestamp.utcnow().isoformat()
    cache_path = DATA_CACHE_DIR / f"bg_age_{ACS_YEAR}.csv"
    df.to_csv(cache_path, index=False)

    summary = {
        "timestamp": timestamp,
        "records": len(df),
    }
    cols_to_return = ["block_group_id", "tract_id", "bg_age_5_17", "is_modeled"]
    return df[cols_to_return], summary


@st.cache_data(ttl=86400)
def fetch_tract_enrollment_data() -> Tuple[pd.DataFrame, dict]:
    """Retrieve ACS tract-level enrollment totals and compute K-12 rates."""

    # Enrollment counts from S1401 (kindergarten, grades 1-8, grades 9-12)
    enrollment_params = {
        "get": ",".join(list(TRACT_ENROLLMENT_FIELDS.keys()) + ["NAME"]),
        "for": "tract:*",
        "in": f"state:{STATE_FIPS}+county:{COUNTY_FIPS}",
    }
    enrollment_payload = _acs_request(ACS_SUBJECT_ENDPOINT, enrollment_params)
    enrollment_headers = enrollment_payload[0]
    expected_enrollment = list(TRACT_ENROLLMENT_FIELDS.keys()) + ["NAME", "state", "county", "tract"]
    missing_enrollment = [col for col in expected_enrollment if col not in enrollment_headers]
    if missing_enrollment:
        raise ValueError(
            "Missing S1401 enrollment variables: " + ", ".join(missing_enrollment)
        )

    enrollment_df = pd.DataFrame(enrollment_payload[1:], columns=enrollment_headers)
    enrollment_df["tract_id"] = enrollment_df["state"] + enrollment_df["county"] + enrollment_df["tract"]

    for code, alias in TRACT_ENROLLMENT_FIELDS.items():
        enrollment_df[alias] = pd.to_numeric(enrollment_df[code], errors="coerce")

    enrollment_df["tract_enrolled_k12"] = (
        enrollment_df["kindergarten_total"].fillna(0)
        + enrollment_df["grades_1_to_8_total"].fillna(0)
        + enrollment_df["grades_9_to_12_total"].fillna(0)
    )

    # Tract population age 5-17 from B01001
    tract_params = {
        "get": ",".join(list(BG_AGE_FIELDS.keys()) + ["NAME"]),
        "for": "tract:*",
        "in": f"state:{STATE_FIPS}+county:{COUNTY_FIPS}",
    }
    tract_payload = _acs_request(ACS_ACS5_ENDPOINT, tract_params)
    tract_headers = tract_payload[0]
    expected_tract = list(BG_AGE_FIELDS.keys()) + ["NAME", "state", "county", "tract"]
    missing_tract = [col for col in expected_tract if col not in tract_headers]
    if missing_tract:
        raise ValueError(f"Missing tract B01001 variables: {', '.join(missing_tract)}")

    tract_df = pd.DataFrame(tract_payload[1:], columns=tract_headers)
    tract_df["tract_id"] = tract_df["state"] + tract_df["county"] + tract_df["tract"]
    for code, alias in BG_AGE_FIELDS.items():
        tract_df[alias] = pd.to_numeric(tract_df[code], errors="coerce")
    tract_df["tract_pop_5_17"] = tract_df[list(BG_AGE_FIELDS.values())].fillna(0).sum(axis=1)

    combined = enrollment_df.merge(tract_df[["tract_id", "tract_pop_5_17"]], on="tract_id", how="left")
    combined["tract_pop_5_17"] = combined["tract_pop_5_17"].fillna(0)
    combined["tract_rate_k12"] = combined.apply(
        lambda row: 0.0 if row["tract_pop_5_17"] <= 0 else row["tract_enrolled_k12"] / row["tract_pop_5_17"],
        axis=1
    )

    timestamp = pd.Timestamp.utcnow().isoformat()
    cache_path = DATA_CACHE_DIR / f"tract_enrollment_{ACS_YEAR}.csv"
    combined.to_csv(cache_path, index=False)

    summary = {
        "timestamp": timestamp,
        "records": len(combined),
    }

    cols = [
        "tract_id",
        "tract_enrolled_k12",
        "tract_pop_5_17",
        "tract_rate_k12",
    ]
    return combined[cols], summary


def validate_k12_total(demographics: pd.DataFrame, tract_enrollment_data: pd.DataFrame) -> dict:
    """
    Validate K-12 total against expected range (150k-260k).
    Return validation result with informational notices if out of range.
    """
    total = float(demographics['k12_pop'].sum())
    min_expected = 150000.0  # Adjusted from 200k to 150k based on ACS 2023 enrollment data
    max_expected = 260000.0
    
    result = {
        'total': total,
        'is_valid': min_expected <= total <= max_expected,
        'min_expected': min_expected,
        'max_expected': max_expected,
        'notices': []
    }
    
    if total < min_expected:
        result['notices'].append(
            f"ℹ️ K-12 total ({total:,.0f}) is below expected minimum ({min_expected:,.0f}). This reflects enrollment vs. population age 5-17."
        )
        # Show top 5 tracts by lowest rate
        tract_enrollment_data_sorted = tract_enrollment_data.sort_values('tract_rate_k12').head(5)
        result['low_tracts'] = tract_enrollment_data_sorted[['tract_id', 'tract_rate_k12', 'tract_enrolled_k12', 'tract_pop_5_17']].copy()
    
    elif total > max_expected:
        result['notices'].append(
            f"ℹ️ K-12 total ({total:,.0f}) exceeds expected maximum ({max_expected:,.0f})."
        )
        # Show top 5 tracts by highest rate
        tract_enrollment_data_sorted = tract_enrollment_data.sort_values('tract_rate_k12', ascending=False).head(5)
        result['high_tracts'] = tract_enrollment_data_sorted[['tract_id', 'tract_rate_k12', 'tract_enrolled_k12', 'tract_pop_5_17']].copy()
    
    else:
        result['messages'] = [f"✅ K-12 total ({total:,.0f}) is within expected range ({min_expected:,.0f}–{max_expected:,.0f})."]
    
    return result


def compute_edi_hpfi_zones(demographics: pd.DataFrame, edi_col: str = "EDI", hpfi_col: str = "hpfi") -> pd.DataFrame:
    """
    Create 4-zone overlay using 75th percentile thresholds for EDI and HPFI.
    
    Zones:
    - Golden Zone: high EDI & high HPFI
    - Mission Zone: high EDI & low HPFI
    - Affluent Opportunity Zone: low EDI & high HPFI
    - Low Priority Zone: low EDI & low HPFI
    """
    working = demographics.copy()
    
    # Ensure numeric
    working[edi_col] = pd.to_numeric(working.get(edi_col), errors='coerce').fillna(0)
    working[hpfi_col] = pd.to_numeric(working.get(hpfi_col), errors='coerce').fillna(0)
    
    # Compute quantiles
    edi_75 = working[edi_col].quantile(0.75)
    hpfi_75 = working[hpfi_col].quantile(0.75)
    
    edi_high = working[edi_col] >= edi_75
    hpfi_high = working[hpfi_col] >= hpfi_75
    
    def classify_zone(edi_h, hpfi_h):
        if edi_h and hpfi_h:
            return 'Golden Zone'
        elif edi_h and not hpfi_h:
            return 'Mission Zone'
        elif not edi_h and hpfi_h:
            return 'Affluent Opportunity Zone'
        else:
            return 'Low Priority Zone'
    
    working['zone'] = [classify_zone(e, h) for e, h in zip(edi_high, hpfi_high)]
    
    # Store thresholds as metadata
    working['_edi_75_threshold'] = edi_75
    working['_hpfi_75_threshold'] = hpfi_75
    
    return working


def compute_marketing_zones(demographics: pd.DataFrame, edi_col: str = "EDI", hpfi_col: str = "hpfi") -> pd.DataFrame:
    """
    Create High-Potential Marketing Zones optimized for growth strategy.
    
    Focuses on areas with:
    - High HPFI (tuition-paying potential) 
    - Moderate EDI (some access gaps but not extreme deserts)
    - Strong K-12 population
    
    Zones:
    - Premium Growth: High HPFI (≥75th), Moderate EDI (25-75th), High K12
    - Established Markets: High HPFI (≥75th), Low EDI (<25th) - affluent, well-served
    - Emerging Opportunity: Medium HPFI (50-75th), Moderate EDI (25-75th)
    - Foundation Building: All other combinations
    """
    working = demographics.copy()
    
    # Ensure numeric
    working[edi_col] = pd.to_numeric(working.get(edi_col), errors='coerce').fillna(0)
    working[hpfi_col] = pd.to_numeric(working.get(hpfi_col), errors='coerce').fillna(0)
    working['k12_pop'] = pd.to_numeric(working.get('k12_pop'), errors='coerce').fillna(0)
    
    # Compute quantiles
    edi_25 = working[edi_col].quantile(0.25)
    edi_75 = working[edi_col].quantile(0.75)
    hpfi_50 = working[hpfi_col].quantile(0.50)
    hpfi_75 = working[hpfi_col].quantile(0.75)
    k12_median = working['k12_pop'].median()
    
    def classify_marketing_zone(row):
        edi = row[edi_col]
        hpfi = row[hpfi_col]
        k12 = row['k12_pop']
        
        # Premium Growth: High HPFI + Moderate EDI + Good K12 population
        if hpfi >= hpfi_75 and edi_25 <= edi <= edi_75 and k12 >= k12_median:
            return 'Premium Growth Target'
        
        # Established Markets: High HPFI + Low EDI (already well-served but affluent)
        elif hpfi >= hpfi_75 and edi < edi_25:
            return 'Established Market'
        
        # Emerging Opportunity: Medium HPFI + Moderate EDI
        elif hpfi >= hpfi_50 and edi_25 <= edi <= edi_75:
            return 'Emerging Opportunity'
        
        # Foundation Building: Everything else
        else:
            return 'Foundation Building'
    
    working['marketing_zone'] = working.apply(classify_marketing_zone, axis=1)
    
    # Store thresholds as metadata
    working['_edi_25_threshold'] = edi_25
    working['_edi_75_threshold'] = edi_75
    working['_hpfi_50_threshold'] = hpfi_50
    working['_hpfi_75_threshold'] = hpfi_75
    working['_k12_median'] = k12_median
    
    return working


# Page config
st.set_page_config(
    page_title="CCA Growth Opportunity Explorer", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS for professional blue/grey theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    .main > div {
        padding-top: 2rem;
        background-color: #F8F9FA;
    }
    
    .stMetric {
        background-color: #ffffff;
        border: 2px solid #0070C0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background-color: #0070C0;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #005A9C;
    }
    
    h1, h2, h3 {
        color: #4F4F4F;
    }
    
    .stInfo {
        background-color: #E3F2FD;
        border-left: 4px solid #0070C0;
        border-radius: 4px;
    }
    
    .stExpander {
        border: 1px solid #0070C0;
        border-radius: 6px;
    }
    
    /* Remove red styling, use blue info boxes */
    .stAlert {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


def compute_hpfi_scores(df: pd.DataFrame, edi_col: str = "EDI") -> pd.DataFrame:
    """Attach High-Potential Family Index (0-1) to the provided DataFrame.
    
    Tuned to prioritize tuition-paying potential through higher income weighting,
    inverse poverty signal, and proximity to CCA campuses to support growth strategy.
    """

    def normalise(series: pd.Series) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce")
        if series.nunique(dropna=True) <= 1:
            return pd.Series(0.5, index=series.index)
        scaled = MinMaxScaler().fit_transform(series.to_frame()).flatten()
        return pd.Series(scaled, index=series.index)

    working = df.copy()
    
    # CCA Campus locations
    cca_campuses = pd.DataFrame({
        'name': ['West Oak Lane', 'Hunting Park'],
        'lat': [40.056339, 40.015278],
        'lon': [-75.153858, -75.138889]
    })

    candidate_names = []
    if isinstance(edi_col, str):
        candidate_names.extend([edi_col, edi_col.lower(), edi_col.upper()])
    candidate_names.extend([col for col in working.columns if isinstance(col, str) and col.lower() == str(edi_col).lower()])

    edi_series = None
    for name in candidate_names:
        if isinstance(name, str) and name in working.columns:
            maybe_series = working[name]
            edi_series = maybe_series if isinstance(maybe_series, pd.Series) else pd.Series(maybe_series)
            break

    if edi_series is None:
        edi_series = pd.Series(np.nan, index=working.index)
    else:
        if not isinstance(edi_series, pd.Series):
            edi_series = pd.Series(edi_series)
        if len(edi_series.index) != len(working.index):
            edi_series = edi_series.reset_index(drop=True)
            edi_series = edi_series.reindex(range(len(working.index)))
        edi_series.index = working.index

    income_norm = normalise(working.get("income"))
    k12_norm = normalise(working.get("k12_pop"))

    # Inverse poverty as additional tuition-potential signal
    poverty_series = pd.to_numeric(working.get("poverty_rate"), errors="coerce").fillna(0)
    inverse_poverty = 1.0 - (poverty_series / 100.0).clip(lower=0.0, upper=1.0)
    inverse_poverty_norm = normalise(pd.Series(inverse_poverty, index=working.index))

    edi_values = pd.to_numeric(edi_series, errors="coerce").fillna(0.0)
    inverse_edi = 1.0 - (edi_values / 100.0).clip(lower=0.0, upper=1.0)
    
    # Campus proximity score (closer = higher HPFI)
    proximity_scores = []
    for _, row in working.iterrows():
        if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
            min_dist = min([
                haversine_km(row['lat'], row['lon'], campus['lat'], campus['lon'])
                for _, campus in cca_campuses.iterrows()
            ])
            # Convert to score: 0km=1.0, 20km=0.0, exponential decay
            proximity_score = np.exp(-min_dist / 8.0)  # 8km half-life
        else:
            proximity_score = 0.0
        proximity_scores.append(proximity_score)
    
    proximity_norm = normalise(pd.Series(proximity_scores, index=working.index))

    # Updated weights including campus proximity
    weights = {
        "income": 0.50,           # Primary tuition signal
        "inverse_poverty": 0.20,  # Economic stability
        "proximity": 0.15,        # NEW - Distance to CCA campuses
        "k12": 0.10,              # Market size
        "inverse_edi": 0.05,      # Low competition
    }

    hpfi = (
        weights["income"] * income_norm +
        weights["inverse_poverty"] * inverse_poverty_norm +
        weights["proximity"] * proximity_norm +
        weights["k12"] * k12_norm +
        weights["inverse_edi"] * inverse_edi
    )

    working["hpfi"] = hpfi.clip(0.0, 1.0)
    working["nearest_campus_km"] = proximity_scores  # Store for reference
    return working

@st.cache_data(ttl=3600)  # Cache for 1 hour, then reload
def load_block_group_data():
    """Load block group geometries and enrich demographics with modeled ACS enrollment."""

    try:
        gdf = gpd.read_file('philadelphia_block_groups.geojson')
        demographics = pd.read_csv('demographics_block_groups.csv')
    except Exception as exc:
        st.error(f"Error loading block group data: {exc}")
        st.info("Please run fetch_block_groups.py first to download Census block group data")
        st.stop()

    keep_cols = [col for col in ['GEOID', 'geometry'] if col in gdf.columns]
    gdf = gdf[keep_cols].copy()

    gdf['GEOID'] = gdf['GEOID'].astype(str)
    demographics['block_group_id'] = demographics['block_group_id'].astype(str)

    sentinel_cols = ['income', 'poverty_rate', 'total_pop', 'pct_black', 'pct_white', 'hh_with_u18', '%Christian']
    for col in sentinel_cols:
        if col in demographics.columns:
            demographics[col] = demographics[col].replace(-666666666, pd.NA)
            demographics[col] = pd.to_numeric(demographics[col], errors='coerce')

    if 'TRACTCE' in demographics.columns:
        demographics['TRACTCE'] = demographics['TRACTCE'].astype(str)

    legacy_total = pd.to_numeric(demographics.get('k12_pop'), errors='coerce').sum()
    if 'k12_pop' in demographics.columns:
        demographics.rename(columns={'k12_pop': 'k12_pop_legacy'}, inplace=True)
    else:
        demographics['k12_pop_legacy'] = pd.NA

    try:
        bg_age_df, bg_meta = fetch_block_group_age_data()
        tract_enrollment_df, tract_meta = fetch_tract_enrollment_data()
    except Exception as exc:
        raise RuntimeError(
            "Unable to retrieve ACS 2023 B01001/S1401 data. Provide a valid CENSUS_API_KEY and internet access."
        ) from exc

    demographics = demographics.merge(bg_age_df, on='block_group_id', how='left')
    demographics['tract_id'] = demographics['block_group_id'].str.slice(0, 11)
    demographics = demographics.merge(tract_enrollment_df, on='tract_id', how='left')

    missing_mask = demographics['tract_rate_k12'].isna() | demographics['bg_age_5_17'].isna()
    demographics['bg_age_5_17'] = demographics['bg_age_5_17'].fillna(0)
    demographics['tract_rate_k12'] = demographics['tract_rate_k12'].fillna(0)
    demographics['bg_enrolled_k12'] = (demographics['bg_age_5_17'] * demographics['tract_rate_k12']).clip(lower=0)
    demographics['k12_pop'] = demographics['bg_enrolled_k12'].round(0)
    demographics['is_modeled'] = True
    demographics['k12_imputed'] = False

    if missing_mask.any():
        demographics.loc[missing_mask, 'k12_pop'] = 0
        demographics.loc[missing_mask, 'k12_imputed'] = True

    demos_summary = {
        'block_groups': len(demographics),
        'k12_total': float(demographics['k12_pop'].sum()),
        'legacy_k12_total': float(legacy_total) if pd.notna(legacy_total) else None,
        'imputed_count': int(demographics['k12_imputed'].sum()),
        'bg_cache_timestamp': bg_meta.get('timestamp'),
        'tract_cache_timestamp': tract_meta.get('timestamp'),
    }

    print(
        f"[QA] Block groups loaded: {demos_summary['block_groups']}. "
        f"ACS {ACS_YEAR} modeled K-12: {demos_summary['k12_total']:,.0f}. "
        f"Imputed block groups: {demos_summary['imputed_count']}."
    )

    return gdf, demographics, demos_summary

@st.cache_data  
def load_current_students():
    """Load current student overlay data (anonymized locations)."""
    try:
        return pd.read_csv('current_students_anonymized.csv')
    except Exception as e:
        st.error(f"Error loading student overlay data: {e}")
        return pd.DataFrame()

def create_choropleth_map(
    gdf_filtered,
    demographics_filtered,
    color_column,
    title,
    show_students=False,
    students_df=None,
    show_competition=False,
    competition_df=None,
    use_zones=False
):
    """Create choropleth map with block group boundaries.
    
    Args:
        use_zones: If True, color by 'zone' column with 4-category scheme.
    """
    
    # Debug output
    st.sidebar.write(f"🔍 Debug: Creating map with {len(gdf_filtered)} geographic features")
    st.sidebar.write(f"🔍 Debug: Demographic data has {len(demographics_filtered)} records")
    
    # Merge geodata with demographic data
    plot_data = gdf_filtered.merge(demographics_filtered, left_on='GEOID', right_on='block_group_id', how='left')

    if 'EDI' in plot_data.columns:
        plot_data['EDI'] = pd.to_numeric(plot_data['EDI'], errors='coerce')
    if 'hpfi' in plot_data.columns:
        plot_data['hpfi'] = pd.to_numeric(plot_data['hpfi'], errors='coerce')
    if 'k12_pop' in plot_data.columns:
        plot_data['k12_pop'] = pd.to_numeric(plot_data['k12_pop'], errors='coerce')
    
    st.sidebar.write(f"🔍 Debug: Merged data has {len(plot_data)} rows")
    
    if len(plot_data) == 0:
        st.error("❌ No data to display on map after filtering!")
        return go.Figure()
    
    # Check if color_column or zone column exists
    if use_zones == 'marketing':
        if 'marketing_zone' not in plot_data.columns:
            st.error("Marketing zone column not found. Cannot create marketing zones map.")
            return go.Figure()
        color_column = 'marketing_zone'
    elif use_zones:
        if 'zone' not in plot_data.columns:
            st.error("Zone column not found. Cannot create overlay map.")
            return go.Figure()
        color_column = 'zone'
    elif color_column not in plot_data.columns:
        st.warning(f"⚠️ Column '{color_column}' not found. Available columns: {list(plot_data.columns)}")
        # Use a default column
        if 'k12_pop' in plot_data.columns:
            color_column = 'k12_pop'
        else:
            return go.Figure()
    
    # Ensure key id column exists after merge
    if 'block_group_id' not in plot_data.columns and 'GEOID' in plot_data.columns:
        plot_data['block_group_id'] = plot_data['GEOID'].astype(str)

    # Prepare GeoJSON
    geojson_data = json.loads(plot_data.to_json())
    
    # Add properties to each feature for hover display
    for idx, feature in enumerate(geojson_data['features']):
        if idx < len(plot_data):
            row = plot_data.iloc[idx]
            feature['id'] = str(idx)
            feature['properties']['block_group_id'] = str(row.get('block_group_id', 'N/A'))
            k12_val = row.get('k12_pop', np.nan)
            income_val = row.get('income', np.nan)
            poverty_val = row.get('poverty_rate', np.nan)
            feature['properties']['k12_pop'] = 0 if pd.isna(k12_val) else int(k12_val)
            feature['properties']['income'] = 0 if pd.isna(income_val) else int(income_val)
            feature['properties']['poverty_rate'] = 0.0 if pd.isna(poverty_val) else float(poverty_val)
            if color_column in row:
                color_val = row.get(color_column, np.nan)
                if use_zones:
                    feature['properties'][color_column] = str(color_val) if pd.notna(color_val) else 'Unknown'
                else:
                    feature['properties'][color_column] = 0.0 if pd.isna(color_val) else float(color_val)
    
    # Build custom hover template with proper formatting for missing data
    hover_template = '<b>Block Group: %{customdata[0]}</b><br>'
    if use_zones:
        hover_template += 'Zone: %{customdata[5]}<br>'
    hover_template += 'EDI: %{customdata[1]}<br>'
    hover_template += 'Median Income: %{customdata[2]}<br>'
    hover_template += 'K-12 Population: %{customdata[3]}<br>'
    hover_template += 'HPFI: %{customdata[4]}<extra></extra>'
    
    # Prepare custom data for hover with proper formatting
    cleaned = plot_data.copy()

    numeric_cols = ['income', 'k12_pop', 'hpfi', 'EDI']
    if not use_zones:
        numeric_cols.append(color_column)
    
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].replace(-666666666, np.nan)
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

    # Handle zone-based coloring
    if use_zones == 'marketing':
        # Marketing zones colorscale
        zone_map = {
            'Premium Growth Target': 0,
            'Established Market': 1,
            'Emerging Opportunity': 2,
            'Foundation Building': 3
        }
        z_vals = cleaned['marketing_zone'].map(zone_map).fillna(3).astype(int).values
        colorscale = [
            [0.0, '#00B050'],     # Premium Growth - green
            [0.33, '#0070C0'],    # Established Market - blue
            [0.67, '#FFC000'],    # Emerging Opportunity - gold
            [1.0, '#A0A0A0']      # Foundation Building - grey
        ]
    elif use_zones:
        # EDI×HPFI overlay zones
        zone_map = {
            'Golden Zone': 0,
            'Mission Zone': 1,
            'Affluent Opportunity Zone': 2,
            'Low Priority Zone': 3
        }
        z_vals = cleaned['zone'].map(zone_map).fillna(3).astype(int).values
        colorscale = [
            [0.0, '#00B050'],     # Golden Zone - green
            [0.33, '#FFC000'],    # Mission Zone - gold
            [0.67, '#0070C0'],    # Affluent Opportunity - blue
            [1.0, '#C00000']      # Low Priority - red
        ]
    else:
        z_series = cleaned[color_column] if color_column in cleaned.columns else pd.Series(np.nan, index=cleaned.index)
        if z_series.isna().all():
            fallback_col = 'k12_pop' if 'k12_pop' in cleaned.columns else None
            if fallback_col:
                st.warning(f"No valid values in '{color_column}' to color by; falling back to '{fallback_col}'.")
                color_column = fallback_col
                if color_column in cleaned.columns:
                    z_series = pd.to_numeric(cleaned[color_column], errors='coerce')
            else:
                st.error("No valid data available for coloring the map.")
                return go.Figure()
        z_vals = z_series.fillna(0).astype(float).values
        colorscale = "RdYlBu_r"

    # Helper function to format values for display
    def format_value(row, col, format_type='number'):
        """Format value for hover display, showing 'N/A' for missing data"""
        val = row.get(col, np.nan)
        if pd.isna(val):
            return 'N/A'
        if format_type == 'currency':
            return f'${val:,.0f}'
        elif format_type == 'percent':
            return f'{val:.1f}%'
        elif format_type == 'integer':
            return f'{int(val):,}'
        elif format_type == 'hpfi':
            return f'{val:.2f}'
        else:
            return f'{val:.1f}'

    # Create formatted customdata
    customdata_list = []
    for _, row in cleaned.iterrows():
        zone_val = str(row.get('zone', 'Unknown')) if use_zones else ''
        customdata_list.append([
            str(row.get('block_group_id', 'N/A')),
            format_value(row, 'EDI', 'number'),
            format_value(row, 'income', 'currency'),
            format_value(row, 'k12_pop', 'integer'),
            format_value(row, 'hpfi', 'hpfi'),
            zone_val,
        ])

    customdata = np.array(customdata_list)
    
    # Create the choropleth using graph_objects for better control
    fig = go.Figure()
    
    # Prepare colorbar based on mode
    if use_zones == 'marketing':
        colorbar = dict(
            title='Marketing Zone',
            tickvals=[0, 1, 2, 3],
            ticktext=['Premium Growth', 'Established', 'Emerging', 'Foundation'],
            len=0.7,
            x=1.02
        )
    elif use_zones:
        colorbar = dict(
            title='Zone',
            tickvals=[0, 1, 2, 3],
            ticktext=['Golden', 'Mission', 'Affluent', 'Low Priority'],
            len=0.7,
            x=1.02
        )
    else:
        colorbar = dict(
            title=color_column.replace('_', ' ').title(),
            len=0.7,
            x=1.02
        )
    
    # Add choropleth trace
    fig.add_choroplethmapbox(
        geojson=geojson_data,
        locations=plot_data.index,
        z=z_vals,
        featureidkey="id",  # Link to the 'id' we set in the GeoJSON features
        customdata=customdata,
        colorscale=colorscale,
        marker_opacity=0.7,
        marker_line_width=1,
        marker_line_color='white',
        hovertemplate=hover_template,
        colorbar=colorbar,
        showscale=True
    )

    # Compute bounds and center for auto-fit
    try:
        # Ensure CRS is WGS84
        if gdf_filtered.crs is None or gdf_filtered.crs.to_string() != "EPSG:4326":
            gdf_plot = gdf_filtered.to_crs(epsg=4326)
        else:
            gdf_plot = gdf_filtered
        
        # Only use bounds if we have valid data
        if len(gdf_plot) > 0:
            minx, miny, maxx, maxy = gdf_plot.total_bounds
            # Sanity check: Philadelphia bounds should be around -75.28 to -74.95 lon, 39.87 to 40.14 lat
            if -76 < minx < -74 and -76 < maxx < -74 and 39 < miny < 41 and 39 < maxy < 41:
                center_lat = (miny + maxy) / 2
                center_lon = (minx + maxx) / 2
                # Calculate zoom based on span
                lat_span = maxy - miny
                lon_span = maxx - minx
                max_span = max(lat_span, lon_span)
                zoom = 11 if max_span > 0.2 else 12
            else:
                # Bounds are invalid, use Philadelphia defaults
                center_lat, center_lon = 39.952, -75.193
                zoom = 11
        else:
            center_lat, center_lon = 39.952, -75.193
            zoom = 11
    except Exception:
        center_lat, center_lon = 39.952, -75.193
        zoom = 11

    # Update layout for map with auto-fit to locations
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=zoom,
        title=title,
        height=700,
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    
    # Add CCA campus markers (yellow stars with address)
    cca_campuses = pd.DataFrame({
        'name': ['CCA Main Campus (58th St)', 'CCA Baltimore Ave Campus'],
        'lat': [39.9386, 39.9508],
        'lon': [-75.2312, -75.2085],
        'address': ['1939 S. 58th St. Philadelphia, PA', '4109 Baltimore Ave Philadelphia, PA']
    })
    
    fig.add_scattermapbox(
        lat=cca_campuses['lat'],
        lon=cca_campuses['lon'],
        mode='markers',
        marker=dict(size=16, color='gold', symbol='star'),
        text=cca_campuses['name'] + ' — ' + cca_campuses['address'],
        name='CCA Campuses',
        hovertemplate='<b>🏫 %{text}</b><br>' +
                     'Lat: %{lat:.4f}<br>' +
                     'Lon: %{lon:.4f}<extra></extra>'
    )
    
    # Add current students if requested
    if show_students and students_df is not None and not students_df.empty:
        fig.add_scattermapbox(
            lat=students_df['lat'],
            lon=students_df['lon'],
            mode='markers',
            marker=dict(size=6, color='blue', opacity=0.6),
            name='Current Students',
            hovertemplate='<b>👨‍🎓 Current CCA Student</b><br>' +
                         'Lat: %{lat:.4f}<br>' +
                         'Lon: %{lon:.4f}<extra></extra>'
        )

    if show_competition and competition_df is not None and not competition_df.empty:
        palette = {
            'Catholic': '#8E0000',
            'Charter': '#1F77B4',
            'Private': '#7851A9',
            'Christian': '#FF7F0E',
            'CCA Campus': '#FFD700',
            'Other': '#2CA02C'
        }
        comp_copy = competition_df.copy()
        comp_copy['type'] = comp_copy['type'].fillna('Other').astype(str)
        # Map colors based on school type - ensure we get a proper list
        comp_copy['color'] = comp_copy['type'].apply(lambda t: palette.get(t, '#2CA02C'))
        comp_copy['capacity'] = pd.to_numeric(comp_copy.get('capacity'), errors='coerce')

        def grade_band(label: object) -> str:
            if pd.isna(label):
                return 'Unknown'
            text = str(label).upper()
            if 'K-12' in text:
                return 'K-12'
            if 'PK-12' in text:
                return 'PK-12'
            if 'PK-8' in text or 'K-8' in text:
                return 'K-8'
            if '9-12' in text:
                return '9-12'
            if '6-12' in text:
                return '6-12'
            return text.title()

        def format_tuition(value: object) -> str:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return 'N/A'
            if isinstance(value, str) and value.strip() == '':
                return 'N/A'
            numeric = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
            if pd.isna(numeric):
                return str(value)
            return f"${numeric:,.0f}"

        comp_copy['grade_band'] = comp_copy.get('grades').apply(grade_band) if 'grades' in comp_copy.columns else 'Unknown'
        if 'tuition' in comp_copy.columns:
            comp_copy['tuition_display'] = comp_copy['tuition'].apply(format_tuition)
        else:
            comp_copy['tuition_display'] = 'N/A'

        comp_copy['capacity_display'] = comp_copy['capacity'].apply(
            lambda v: f"{int(v):,}" if pd.notna(v) else 'n/a'
        )
        comp_copy['notes'] = comp_copy.get('notable_info', '').fillna('')

        hover_cols = ['school_name', 'type', 'grade_band', 'tuition_display', 'capacity_display', 'address', 'notes']
        custom_comp = comp_copy[hover_cols].fillna('').to_numpy()

        # Ensure no NaN values in coordinates or colors
        valid_idx = comp_copy['lat'].notna() & comp_copy['lon'].notna() & comp_copy['color'].notna()
        if valid_idx.sum() > 0:
            comp_valid = comp_copy[valid_idx].copy()
            custom_valid = custom_comp[valid_idx.values]
            
            fig.add_scattermapbox(
                lat=comp_valid['lat'].tolist(),
                lon=comp_valid['lon'].tolist(),
                mode='markers',
                marker=dict(
                    size=14,
                    symbol='diamond',
                    color=comp_valid['color'].tolist(),
                    opacity=0.95
                ),
                customdata=custom_valid,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Type: %{customdata[1]}<br>"
                    "Grades: %{customdata[2]}<br>"
                    "Tuition: %{customdata[3]}<br>"
                    "Seats: %{customdata[4]}<br>"
                    "Address: %{customdata[5]}<br>"
                    "%{customdata[6]}<extra></extra>"
                ),
                name='Competitor Schools'
            )
    
    return fig

def calculate_marketing_priority_bg(demographics_df, cca_campuses):
    """Calculate marketing priority scores for block groups"""
    
    marketing_scores = []
    
    for _, bg in demographics_df.iterrows():
        score = 0
        
        # Distance to nearest CCA campus (highest weight)
        distances = []
        for _, campus in cca_campuses.iterrows():
            dist = haversine_km(bg['lat'], bg['lon'], campus['lat'], campus['lon'])
            distances.append(dist)
        
        min_distance = min(distances) if distances else 50
        
        # Distance scoring (closer is better)
        if min_distance <= 2:
            score += 4
        elif min_distance <= 5:
            score += 3
        elif min_distance <= 10:
            score += 2
        elif min_distance <= 20:
            score += 1
        
        # Income scoring (target range: $50K-$350K)
        income = bg.get('income', 0)
        if 100000 <= income <= 350000:
            score += 3  # Premium tier
        elif 75000 <= income < 100000:
            score += 2  # High income
        elif 50000 <= income < 75000:
            score += 1  # Upper middle
        
        # Christian percentage (alignment with mission)
        christian_pct = bg.get('%Christian', 0)
        if christian_pct >= 30:
            score += 1
        
        # K-12 population (market size)
        k12_pop = bg.get('k12_pop', 0)
        if k12_pop >= 200:
            score += 1
        
        marketing_scores.append(score)
    
    return marketing_scores

# Main app
def main():
    st.title("� Cornerstone Christian Academy Growth Opportunity Explorer")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0070C0 0%, #005A9C 100%); padding: 1.5rem; border-radius: 8px; color: white; margin-bottom: 1rem;'>
        <h3 style='color: white; margin: 0;'>Informative Insights for Strategic Outreach & Inclusive Expansion</h3>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Identify high-potential areas to broaden CCA's reach across diverse socioeconomic groups and support balanced, inclusive growth. Data from ACS 2023.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add explanation of EDI in an expandable section
    with st.expander("ℹ️ Understanding the Educational Desert Index (EDI)"):
        st.markdown("""
        ### What EDI Measures

        The Educational Desert Index uses **rigorous 2-Step Floating Catchment Area (2SFCA) analysis** to identify neighborhoods with limited access to quality educational infrastructure. This is the same methodology used in academic research for healthcare access and food deserts.

        **Higher EDI scores = worse educational environment** (true educational deserts).

        #### How We Calculate It (2SFCA Method)
        1. **Accessibility (40%)** – Uses gravity-weighted school access model. Nearby schools count more than distant ones due to exponential distance decay (β=5km). Measures actual capacity availability, not just proximity.
        2. **School-to-Student Ratio (30%)** – Local capacity vs K-12 population using gravity-weighted catchment areas. Accounts for overcrowding and shared demand.
        3. **Socioeconomic Need (20%)** – Combines poverty rate (70%) and % adults without HS diploma (30%). Real barriers to educational access.
        4. **Infrastructure (10%)** – Estimated broadband access using poverty as proxy (lower poverty areas have ~85-95% access, higher poverty ~60-70%). Future versions will use ACS S2801 broadband data when available.

        Each component is normalized 0–1 and weighted, then scaled to 0–100.

        #### Why 2SFCA is Better
        - **Prevents edge effects**: Distant mega-schools don't dominate scores
        - **Gravity decay**: Schools 1km away count far more than schools 10km away
        - **True crowding**: Measures seats-per-weighted-student, not simple ratios
        - **Research-validated**: Standard method in spatial accessibility analysis

        #### Reading the Score
        - **70–100** → True educational deserts: far from schools, overcrowded, high poverty, poor infrastructure
        - **40–69** → Moderate challenges: some gaps in access or capacity
        - **0–39** → Well-served: good school access, sufficient seats, better infrastructure

        Use EDI to identify underserved areas where CCA can fill real educational infrastructure gaps.
        """)
    
    # Load data
    with st.spinner("Loading Census block group data..."):
        try:
            gdf, demographics, k12_summary = load_block_group_data()
            # Also fetch tract data for validation
            tract_enrollment_data, _ = fetch_tract_enrollment_data()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        current_students = load_current_students()

    # Validate K-12 total
    validation_result = validate_k12_total(demographics, tract_enrollment_data)
    
    if not validation_result['is_valid']:
        st.info("📊 K-12 Enrollment Data Notice")
        st.write(validation_result['notices'][0])
        if 'low_tracts' in validation_result:
            st.write("**Tracts with lowest enrollment rates:**")
            st.dataframe(validation_result['low_tracts'], use_container_width=True)
        elif 'high_tracts' in validation_result:
            st.write("**Tracts with highest enrollment rates:**")
            st.dataframe(validation_result['high_tracts'], use_container_width=True)
    else:
        if validation_result.get('messages'):
            st.success(validation_result['messages'][0])
    
    total_students_loaded = k12_summary.get('k12_total', float('nan'))
    total_block_groups_loaded = k12_summary.get('block_groups', len(demographics))
    imputed_count = k12_summary.get('imputed_count', 0)
    component_gaps = k12_summary.get('component_gaps', 0)
    legacy_total = k12_summary.get('legacy_k12_total')
    bg_cache_ts = k12_summary.get('bg_cache_timestamp', 'N/A')
    tract_cache_ts = k12_summary.get('tract_cache_timestamp', 'N/A')

    # Load school data
    try:
        competition_schools = load_competition_schools()
    except RuntimeError as exc:
        st.error(f"Unable to load competition schools: {exc}")
        competition_schools = pd.DataFrame()

    try:
        census_schools = load_census_schools()
    except RuntimeError as exc:
        st.error(f"Unable to load Census school landmarks: {exc}")
        census_schools = pd.DataFrame()

    # Prepare school data
    if not competition_schools.empty:
        if 'type' not in competition_schools.columns:
            competition_schools['type'] = 'Other'
        else:
            competition_schools['type'] = competition_schools['type'].fillna('Other')
        competition_schools['capacity'] = pd.to_numeric(competition_schools.get('capacity'), errors='coerce').fillna(350)
        competition_schools['lat'] = pd.to_numeric(competition_schools.get('lat'), errors='coerce')
        competition_schools['lon'] = pd.to_numeric(competition_schools.get('lon'), errors='coerce')
        competition_schools = competition_schools.dropna(subset=['lat', 'lon'])
    else:
        competition_schools = pd.DataFrame(columns=['school_name', 'type', 'lat', 'lon', 'capacity', 'grades', 'notable_info', 'address'])

    if not census_schools.empty:
        census_schools['lat'] = pd.to_numeric(census_schools.get('lat'), errors='coerce')
        census_schools['lon'] = pd.to_numeric(census_schools.get('lon'), errors='coerce')
        census_schools['capacity'] = pd.to_numeric(census_schools.get('capacity'), errors='coerce').fillna(400)
        census_schools = census_schools.dropna(subset=['lat', 'lon'])
    else:
        census_schools = pd.DataFrame(columns=['school_name', 'lat', 'lon', 'capacity'])

    # Calculate seat capacity
    public_capacity = 0.0
    competitor_capacity = 0.0

    if 'capacity' in census_schools.columns and not census_schools.empty:
        public_capacity = pd.to_numeric(census_schools['capacity'], errors='coerce').fillna(0).sum()

    if 'capacity' in competition_schools.columns and not competition_schools.empty:
        competitor_capacity = pd.to_numeric(competition_schools['capacity'], errors='coerce').fillna(0).sum()

    total_students_value = total_students_loaded if not np.isnan(total_students_loaded) else None
    student_per_public_seat = None
    student_per_combined_seat = None
    if total_students_value is not None and public_capacity > 0:
        student_per_public_seat = total_students_value / public_capacity
    if total_students_value is not None and (public_capacity + competitor_capacity) > 0:
        student_per_combined_seat = total_students_value / (public_capacity + competitor_capacity)

    # ==================== SIDEBAR: DATA SUMMARY ====================
    st.sidebar.title("🎯 CCA Growth Dashboard")
    
    with st.sidebar.expander("📊 Market Overview", expanded=True):
        st.metric(
            "Total K-12 Students",
            f"{total_students_loaded:,.0f}",
            help="ACS 2023 modeled enrollment across Philadelphia block groups"
        )
        st.metric(
            "Block Groups Analyzed",
            f"{total_block_groups_loaded:,}"
        )
        
        if student_per_public_seat is not None:
            st.metric(
                "Students per Public Seat",
                f"{student_per_public_seat:.1f}:1",
                help=f"{public_capacity:,.0f} public school seats available"
            )
        
        st.caption(f"📅 Data: ACS 2023 | Updated {bg_cache_ts[:10]}")
    
    # Data Quality Details (collapsible)
    with st.sidebar.expander("📋 Data Quality & Technical Details"):
        st.write(f"""
        **Data Validation:**
        - Block groups with imputed K-12: {imputed_count:,}
        - Cache timestamps: BG={bg_cache_ts[:10]}, Tract={tract_cache_ts[:10]}
        """)
        
        if legacy_total and not np.isnan(legacy_total):
            st.caption(f"Legacy CSV K-12 total (pre-ACS refresh): {legacy_total:,.0f}")
        
        st.write(f"""
        **School Capacity:**
        - Public seats: {public_capacity:,.0f}
        - Competitor seats: {competitor_capacity:,.0f}
        - Combined seats: {public_capacity + competitor_capacity:,.0f}
        """)
        
        if student_per_combined_seat is not None:
            st.caption(f"Students per seat (with competitors): {student_per_combined_seat:.2f}:1")
        
        st.write(f"""
        **School Data Loaded:**
        - Census public schools: {len(census_schools):,}
        - Competition schools: {len(competition_schools):,}
        """)
    
    # ==================== SIDEBAR: FILTERS ====================
    st.sidebar.divider()
    st.sidebar.header("🎯 Target Area Filters")
    
    # Geographic Filter
    with st.sidebar.expander("📍 **Geographic Scope**", expanded=True):
        max_distance = st.slider(
            "Distance from CCA Campuses (km)",
            min_value=1,
            max_value=35,
            value=15,
            step=1,
            help="Focus on block groups within this distance of CCA campuses",
            key="distance_slider"
        )
        st.caption(f"✓ Showing areas within {max_distance}km of campuses")
    
    # CCA Campus locations
    cca_campuses = pd.DataFrame({
        'name': ['CCA Main Campus (58th St)', 'CCA Baltimore Ave Campus'],
        'lat': [39.9386, 39.9508],
        'lon': [-75.2312, -75.2085],
        'address': ['1939 S. 58th St. Philadelphia, PA', '4109 Baltimore Ave Philadelphia, PA']
    })
    
    # Filter demographics by distance
    def is_within_distance(row):
        for _, campus in cca_campuses.iterrows():
            dist = haversine_km(row['lat'], row['lon'], campus['lat'], campus['lon'])
            if dist <= max_distance:
                return True
        return False
    
    demographics_filtered = demographics[demographics.apply(is_within_distance, axis=1)]
    
    # Optional live data refresh (Census API)
    census_api_key = get_census_api_key()
    if fetch_live_block_groups and census_api_key:
        with st.sidebar.expander("🔄 Live Census Data Refresh"):
            st.caption("Pull fresh ACS data from Census API")
            if st.button("Refresh From Census API", key="refresh_census"):
                with st.spinner("Pulling live ACS data (may take ~30s)..."):
                    try:
                        fetch_live_block_groups()
                        st.success("Live data downloaded. Please rerun the app to see updates.")
                    except Exception as e:
                        st.error(f"Live fetch failed: {e}")
    
    highlight_hpfi = False

    if not demographics_filtered.empty:
        # Income Filter
        with st.sidebar.expander("💰 **Income Targeting**", expanded=False):
            income_series = pd.to_numeric(demographics_filtered['income'], errors='coerce').dropna()
            if income_series.empty:
                income_floor, income_ceiling = 0, 250000
            else:
                income_floor = int(income_series.min())
                income_ceiling = int(income_series.max())
                if income_floor == income_ceiling:
                    income_ceiling = income_floor + 1

            income_range = st.slider(
                "Median Household Income Range",
                min_value=income_floor,
                max_value=income_ceiling,
                value=(income_floor, income_ceiling),
                step=1000,
                format="$%d",
                help="Target areas by economic capacity"
            )
            demographics_filtered = demographics_filtered[
                demographics_filtered['income'].between(income_range[0], income_range[1], inclusive="both")
            ]
            st.caption(f"✓ Showing incomes from ${income_range[0]:,} to ${income_range[1]:,}")

        # HPFI Focus
        with st.sidebar.expander("🎯 **High-Potential Focus**", expanded=False):
            highlight_hpfi = st.checkbox(
                "Show Only High HPFI Areas (≥ 0.75)", 
                value=False,
                help="Focus on top-quartile areas with strongest tuition-paying potential"
            )
            if highlight_hpfi:
                st.caption("✓ Filtering to premium growth opportunities")

        # Refinement Options
        with st.sidebar.expander("🔧 **Refinement Options**", expanded=False):
            # Filter out non-residential block groups
            hide_zero_pop = st.checkbox(
                "Exclude Non-Residential Areas", 
                value=True,
                help="Removes parks, industrial zones with 0 population"
            )
            if hide_zero_pop:
                before_count = len(demographics_filtered)
                demographics_filtered = demographics_filtered[demographics_filtered['total_pop'] > 0].copy()
                removed = before_count - len(demographics_filtered)
                if removed > 0:
                    st.caption(f"✓ Filtered out {removed} non-residential blocks")
            
            # Additional filter for 0 K-12 population
            hide_zero_k12 = st.checkbox(
                "Exclude Areas with 0 K-12 Children",
                value=False,
                help="Focus exclusively on areas with school-age population"
            )
            if hide_zero_k12:
                before_count = len(demographics_filtered)
                demographics_filtered = demographics_filtered[demographics_filtered['k12_pop'] > 0].copy()
                removed = before_count - len(demographics_filtered)
                if removed > 0:
                    st.caption(f"✓ Filtered out {removed} blocks with 0 K-12 children")
            
            # Poverty rate filter - now optional and off by default
            apply_poverty_filter = st.checkbox("Apply Economic Opportunity Filter", value=False)
            if apply_poverty_filter:
                valid_poverty = demographics_filtered['poverty_rate'].dropna()
                if len(valid_poverty) > 0:
                    poverty_range = st.slider(
                        "Economic Stability Range (inverse poverty %)", 
                        float(valid_poverty.min()), 
                        float(valid_poverty.max()), 
                        (float(valid_poverty.min()), float(valid_poverty.max())),
                        step=1.0,
                        help="Lower poverty rates may indicate greater economic capacity"
                    )
                    demographics_filtered = demographics_filtered[
                        (demographics_filtered['poverty_rate'] >= poverty_range[0]) &
                        (demographics_filtered['poverty_rate'] <= poverty_range[1])
                    ]
                else:
                    st.warning("No valid economic data available")
        
    else:
        st.info("📍 No block groups found within the specified radius. Try expanding your geographic scope.")
        demographics_filtered = demographics.iloc[0:0]
        
    # Map Layer Controls
    with st.sidebar.expander("🗺️ **Map Layers**", expanded=False):
        show_current_students = st.checkbox(
            "Show Current Student Locations", 
            value=False,
            help="Overlay current CCA student addresses"
        )
        show_competition_overlay = st.checkbox(
            "Show Other Educational Options", 
            value=True,
            help="Display charter, private, and Catholic schools"
        )
        
        if not competition_schools.empty:
            type_options = sorted(competition_schools['type'].dropna().unique())
        else:
            type_options = []
        selected_competition_types = st.multiselect(
            "School Types to Display",
            options=type_options,
            default=type_options,
            help="Select which types of schools to show"
        ) if type_options else []

        include_competitors_in_edi = st.checkbox(
            "Include Other Schools in EDI Calculation",
            value=False,
            help="Adds charter/private capacity to EDI supply"
        )
        if include_competitors_in_edi:
            st.caption("✓ EDI includes all school seats")
        else:
            st.caption("EDI uses CCA/public seats only")
    
    # Methodology & Help
    with st.sidebar.expander("ℹ️ **About This Dashboard**"):
        st.write("""
        **Purpose:** Identify high-potential growth areas for CCA enrollment.
        
        **Key Metrics:**
        - **HPFI** (High-Potential Family Index): Economic capacity for tuition
        - **EDI** (Educational Desert Index): Access gaps in Christian education
        
        **How to Use:**
        1. Set geographic scope (default 15km from campuses)
        2. Apply income filters to target economic segments
        3. Choose visualization mode (HPFI, EDI, or Overlay)
        4. Explore Top 10 tables for specific block groups
        5. Export filtered data for deeper analysis
        
        **Data Sources:**
        - Census ACS 2023 (demographics, income, K-12 enrollment)
        - NCES school directories (public schools)
        - Local charter/private school data
        
        📧 Questions? Contact your data team.
        """)

    supply_columns = ['lat', 'lon', 'capacity']

    def prepare_supply_frame(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=supply_columns)
        prepared = df.copy()
        for coord_col in ['lat', 'lon']:
            if coord_col in prepared.columns:
                prepared[coord_col] = pd.to_numeric(prepared[coord_col], errors='coerce')
        if 'capacity' not in prepared.columns:
            prepared['capacity'] = 0
        prepared['capacity'] = pd.to_numeric(prepared['capacity'], errors='coerce').fillna(0)
        prepared = prepared.dropna(subset=['lat', 'lon']) if not prepared.empty else prepared
        return prepared[supply_columns]

    census_supply = prepare_supply_frame(census_schools)
    competitor_supply = prepare_supply_frame(competition_schools)

    if include_competitors_in_edi and not competitor_supply.empty:
        edi_supply_df = pd.concat([census_supply, competitor_supply], ignore_index=True)
    else:
        edi_supply_df = census_supply.copy()

    print(
        f"[EDI] Supply rows: {len(edi_supply_df)} | Competitor supply included: {include_competitors_in_edi and not competitor_supply.empty}"
    )
    
    # Visualization selection
    color_options = {
        'High-Potential Family Index (HPFI)': 'hpfi',
        'Educational Desert Index (EDI)': 'EDI',
        'Median Household Income': 'income',
        'First-Generation %': 'first_gen_pct'
    }
    
    map_mode_options = ['HPFI (Tuition Potential)', 'EDI (Access Opportunity)', 'Overlay: EDI × HPFI', 'High-Potential Marketing Zones']
    selected_map_mode = st.sidebar.selectbox(
        "Map Visualization Mode:",
        map_mode_options,
        index=0,
        help="Choose focus: tuition-paying potential (HPFI), access gaps (EDI), combined overlay, or custom marketing zones"
    )
    
    # If overlay mode or marketing zones, use zone colors; otherwise use the classic selection
    if selected_map_mode == 'Overlay: EDI × HPFI':
        selected_metric = 'Overlay: EDI × HPFI'
    elif selected_map_mode == 'High-Potential Marketing Zones':
        selected_metric = 'High-Potential Marketing Zones'
    elif selected_map_mode == 'HPFI (Tuition Potential)':
        selected_metric = 'High-Potential Family Index (HPFI)'
    elif selected_map_mode == 'EDI (Access Opportunity)':
        selected_metric = 'Educational Desert Index (EDI)'
    else:
        selected_metric = st.sidebar.selectbox(
            "Color Map By:", 
            list(color_options.keys()),
            index=0
        )
    
    # Calculate EDI if needed
    if not demographics_filtered.empty:
        if not edi_supply_df.empty:
            with st.spinner("Calculating Educational Desert Index..."):
                try:
                    demographics_with_pop = demographics_filtered[demographics_filtered['total_pop'] > 0].copy()

                    if len(demographics_with_pop) > 0:
                        edi_df = compute_edi_block_groups(demographics_with_pop, edi_supply_df)
                        demographics_filtered = demographics_filtered.merge(
                            edi_df[['block_group_id', 'EDI']],
                            on='block_group_id',
                            how='left'
                        )
                        demographics_filtered = demographics_filtered.copy()
                        demographics_filtered['EDI'] = demographics_filtered['EDI'].fillna(0.0)
                    else:
                        demographics_filtered = demographics_filtered.copy()
                        demographics_filtered['EDI'] = 0.0
                except Exception as e:
                    st.sidebar.warning(f"⚠️ EDI calculation issue: {str(e)[:100]}")
                    demographics_filtered = demographics_filtered.copy()
                    def simple_edi(row):
                        if row.get('total_pop', 0) == 0:
                            return 0.0
                        min_dist = min([
                            haversine_km(row['lat'], row['lon'], c['lat'], c['lon'])
                            for _, c in cca_campuses.iterrows()
                        ])
                        return min(100, min_dist * 5)
                    demographics_filtered['EDI'] = demographics_filtered.apply(simple_edi, axis=1)
        else:
            st.sidebar.info("No school supply data available; using a distance-based EDI estimate.")
            demographics_filtered = demographics_filtered.copy()
            def simple_edi(row):
                if row.get('total_pop', 0) == 0:
                    return 0.0
                min_dist = min([
                    haversine_km(row['lat'], row['lon'], c['lat'], c['lon'])
                    for _, c in cca_campuses.iterrows()
                ])
                return min(100, min_dist * 5)
            demographics_filtered['EDI'] = demographics_filtered.apply(simple_edi, axis=1)

        demographics_filtered = demographics_filtered.copy()
        demographics_filtered['marketing_priority'] = calculate_marketing_priority_bg(demographics_filtered, cca_campuses)

    if 'first_gen_pct' not in demographics_filtered.columns and '%first_gen' in demographics_filtered.columns:
        demographics_filtered['first_gen_pct'] = pd.to_numeric(demographics_filtered['%first_gen'], errors='coerce')
    else:
        demographics_filtered['first_gen_pct'] = pd.to_numeric(demographics_filtered.get('first_gen_pct'), errors='coerce')

    demographics_filtered = compute_hpfi_scores(demographics_filtered, edi_col="EDI" if 'EDI' in demographics_filtered.columns else 'edi')

    # Compute zones if overlay mode or marketing zones mode is selected
    if selected_map_mode == 'Overlay: EDI × HPFI':
        demographics_filtered = compute_edi_hpfi_zones(demographics_filtered, edi_col="EDI", hpfi_col="hpfi")
    elif selected_map_mode == 'High-Potential Marketing Zones':
        demographics_filtered = compute_marketing_zones(demographics_filtered, edi_col="EDI", hpfi_col="hpfi")
    

    hpfi_threshold = None
    if highlight_hpfi and not demographics_filtered.empty:
        hpfi_series = demographics_filtered['hpfi'].dropna()
        if not hpfi_series.empty:
            hpfi_threshold = hpfi_series.quantile(0.75)
            demographics_filtered = demographics_filtered[demographics_filtered['hpfi'] >= hpfi_threshold]
        else:
            st.sidebar.warning("HPFI highlight enabled, but no calculable HPFI values were found.")

    gdf_filtered = gdf[gdf['GEOID'].isin(demographics_filtered['block_group_id'])]
    
    # Main content area
    if not demographics_filtered.empty:
        
        # Key metrics with positive, growth-focused language
        st.markdown("### 📊 Opportunity Snapshot")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Areas in View", 
                len(demographics_filtered),
                help="Number of block groups matching current filters - each represents a potential community for outreach"
            )
        
        with col2:
            total_k12 = int(demographics_filtered['k12_pop'].sum())
            st.metric(
                "Student Population", 
                f"{total_k12:,}",
                help="Total school-age children (K-12) in filtered areas - represents market opportunity"
            )
        
        with col3:
            # Only calculate average from valid (non-NaN) income values
            valid_incomes = demographics_filtered['income'].dropna()
            if len(valid_incomes) > 0:
                avg_income = int(valid_incomes.mean())
                st.metric(
                    "Median Income (Avg)", 
                    f"${avg_income:,}",
                    help="Average of block group median incomes - economic capacity indicator"
                )
            else:
                st.metric(
                    "Median Income (Avg)", 
                    "No data",
                    help="Economic data not available for current selection"
                )
        
        with col4:
            # Show HPFI high-potential count instead of old marketing_priority
            if 'hpfi' in demographics_filtered.columns:
                hpfi_75th = demographics_filtered['hpfi'].quantile(0.75)
                high_hpfi_count = len(demographics_filtered[demographics_filtered['hpfi'] >= hpfi_75th])
                st.metric(
                    "High-Potential Areas", 
                    high_hpfi_count,
                    help=f"Block groups with HPFI ≥ {hpfi_75th:.2f} (top 25% tuition-paying potential)"
                )
            else:
                st.metric(
                    "High-Potential Areas", 
                    "Calculating...",
                    help="HPFI being computed"
                )

        # HPFI detailed breakdown
        if 'hpfi' in demographics_filtered.columns:
            st.markdown("### 💡 Family Potential Index Details")
            st.caption("HPFI measures tuition-paying capacity through income (50%), economic stability (15%), mission alignment (20%), and market size (15%)")
            hpfi_cols = st.columns(3)
            hpfi_avg = demographics_filtered['hpfi'].mean()
            hpfi_top = (demographics_filtered['hpfi'] >= 0.75).sum()
            hpfi_max = demographics_filtered['hpfi'].max()
            with hpfi_cols[0]:
                st.metric("Average HPFI", f"{hpfi_avg:.2f}", help="Mean tuition-paying potential across filtered areas (0.00 = lowest, 1.00 = highest)")
            with hpfi_cols[1]:
                st.metric("Top-Tier Areas (≥0.75)", int(hpfi_top), help="Block groups in top 25% for economic capacity and growth potential")
            with hpfi_cols[2]:
                st.metric("Peak HPFI", f"{hpfi_max:.2f}", help="Highest HPFI score in current selection")
        
        with st.expander("How the Educational Desert Index (EDI) is calculated", expanded=False):
            st.markdown(
                """
                **Rigorous 2-Step Floating Catchment Area (2SFCA) Analysis:**

                **Step 1:** For each school, calculate gravity-weighted demand in its catchment:
                - R_j = seats_j / Σ(pop_i × w(d_ij)) where w(d) = exp(-d/5km)
                
                **Step 2:** For each block group, sum accessibility from all schools:
                - A_i = Σ(R_j × w(d_ij)) = aggregate capacity-to-demand ratio
                
                **Four components (each 0-1, higher = worse):**

                1. **Accessibility (40%)** – Inverse of 2SFCA accessibility score. Accounts for distance decay and competing demand. Lower A_i = higher EDI.
                2. **Seat Ratio (30%)** – Gravity-weighted local seats / K-12 population. Measures true overcrowding with shared catchments.
                3. **Need (20%)** – Poverty rate (70%) + % adults without HS diploma (30%). Socioeconomic barriers to education.
                4. **Infrastructure (10%)** – Estimated broadband access (inverse poverty proxy). Technology access for remote learning.

                **Final EDI** = (0.40×access + 0.30×ratio + 0.20×need + 0.10×infra), scaled 0–100.
                
                **Higher EDI = true educational desert** (poor access, overcrowded, high barriers, low infrastructure).
                
                *2SFCA prevents distant schools from dominating scores and accounts for competing demand across overlapping catchments.*
                """
            )

        # Main map
        st.subheader(f"📍 {selected_metric} by Block Group")
        
        # Check if we're in overlay or marketing zones mode
        is_overlay_mode = selected_metric == 'Overlay: EDI × HPFI'
        is_marketing_zones = selected_metric == 'High-Potential Marketing Zones'
        
        if is_overlay_mode or is_marketing_zones or (selected_metric in color_options and color_options[selected_metric] in demographics_filtered.columns):
            if hpfi_threshold is not None:
                st.info(f"💡 HPFI Focus Mode: Showing block groups with HPFI ≥ {hpfi_threshold:.2f} (top 25% tuition-paying potential).")
            
            # Show zone counts for overlay mode
            if is_overlay_mode and 'zone' in demographics_filtered.columns:
                zone_counts = demographics_filtered['zone'].value_counts()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    golden_count = zone_counts.get('Golden Zone', 0)
                    st.metric('Golden Zone', golden_count, help='High EDI & High HPFI - Priority areas combining access need and economic potential')
                with col2:
                    mission_count = zone_counts.get('Mission Zone', 0)
                    st.metric('Mission Zone', mission_count, help='High EDI & Low HPFI - Mission-focused outreach areas')
                with col3:
                    affluent_count = zone_counts.get('Affluent Opportunity Zone', 0)
                    st.metric('Affluent Opportunity', affluent_count, help='Low EDI & High HPFI - Well-served areas with economic capacity')
                with col4:
                    low_priority_count = zone_counts.get('Low Priority Zone', 0)
                    st.metric('Low Priority Zone', low_priority_count, help='Low EDI & Low HPFI')
            
            # Show marketing zone counts
            if is_marketing_zones and 'marketing_zone' in demographics_filtered.columns:
                st.markdown("### 🎯 Marketing Zone Distribution")
                st.caption("Zones optimized for growth strategy: High HPFI + moderate EDI + strong K-12 population")
                zone_counts = demographics_filtered['marketing_zone'].value_counts()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    premium_count = zone_counts.get('Premium Growth Target', 0)
                    st.metric('🌟 Premium Growth', premium_count, help='High HPFI + Moderate EDI + Strong K12 - Top priority for outreach')
                with col2:
                    established_count = zone_counts.get('Established Market', 0)
                    st.metric('💼 Established Market', established_count, help='High HPFI + Low EDI - Affluent, well-served areas')
                with col3:
                    emerging_count = zone_counts.get('Emerging Opportunity', 0)
                    st.metric('📈 Emerging Opportunity', emerging_count, help='Medium HPFI + Moderate EDI - Growth potential')
                with col4:
                    foundation_count = zone_counts.get('Foundation Building', 0)
                    st.metric('🏗️ Foundation Building', foundation_count, help='Longer-term relationship development areas')
            
            visible_competition = competition_schools[
                competition_schools['type'].isin(selected_competition_types)
            ] if selected_competition_types else pd.DataFrame()
            
            # Determine column name and use_zones flag
            if is_marketing_zones:
                color_col = 'marketing_zone'
                use_zones_flag = 'marketing'
            elif is_overlay_mode:
                color_col = 'zone'
                use_zones_flag = True
            else:
                color_col = color_options.get(selected_metric, 'k12_pop')
                use_zones_flag = False
            
            fig = create_choropleth_map(
                gdf_filtered, 
                demographics_filtered, 
                color_col,
                f"{selected_metric} across Philadelphia Block Groups",
                show_current_students,
                current_students if show_current_students else None,
                show_competition_overlay,
                visible_competition,
                use_zones=use_zones_flag
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info(f"📊 Data for {selected_metric} is being prepared...")
        
        # Analysis tables
        st.subheader("📊 Detailed Analysis")
        
        # Marketing zones mode: show premium growth targets first
        if is_marketing_zones and 'marketing_zone' in demographics_filtered.columns:
            st.write("**Top Premium Growth Targets (High HPFI + Moderate EDI + Strong K12)**")
            premium_zones = demographics_filtered[demographics_filtered['marketing_zone'] == 'Premium Growth Target'].nlargest(10, 'hpfi')[
                ['block_group_id', 'hpfi', 'EDI', 'income', 'k12_pop']
            ].copy()
            if len(premium_zones) > 0:
                premium_zones['hpfi'] = premium_zones['hpfi'].map(lambda v: f"{v:.2f}")
                premium_zones['EDI'] = premium_zones['EDI'].map(lambda v: f"{v:.2f}")
                premium_zones['income'] = premium_zones['income'].map(lambda v: f"${v:,.0f}")
                premium_zones['k12_pop'] = premium_zones['k12_pop'].map(lambda v: f"{int(v):,}")
                st.dataframe(premium_zones, width='stretch', use_container_width=True)
            else:
                st.info("No block groups found in Premium Growth Target zone. Adjust filters to see more areas.")
            st.divider()
        
        # Overlay mode: show zone breakdown first

        if is_overlay_mode and 'zone' in demographics_filtered.columns:
            st.write("**Top Golden Zones (EDI × HPFI Overlay)**")
            golden_zones = demographics_filtered[demographics_filtered['zone'] == 'Golden Zone'].nlargest(10, 'EDI')[
                ['block_group_id', 'EDI', 'hpfi', 'income', 'k12_pop']
            ].copy()
            if len(golden_zones) > 0:
                golden_zones['EDI'] = golden_zones['EDI'].map(lambda v: f"{v:.2f}")
                golden_zones['hpfi'] = golden_zones['hpfi'].map(lambda v: f"{v:.2f}")
                golden_zones['income'] = golden_zones['income'].map(lambda v: f"${v:,.0f}")
                golden_zones['k12_pop'] = golden_zones['k12_pop'].map(lambda v: f"{int(v):,}")
                st.dataframe(golden_zones, width='stretch', use_container_width=True)
            else:
                st.info("No block groups found in Golden Zone.")
            st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Top 10 Block Groups by Educational Desert Index**")
            if 'EDI' in demographics_filtered.columns:
                top_edi = demographics_filtered.nlargest(10, 'EDI')[
                    ['block_group_id', 'EDI', 'k12_pop', 'income']
                ].round(2)
                st.dataframe(top_edi, width='stretch')
        
        with col2:
            st.write("**Top 10 Block Groups by Marketing Priority**")
            if 'marketing_priority' in demographics_filtered.columns:
                top_marketing = demographics_filtered.nlargest(10, 'marketing_priority')[
                    ['block_group_id', 'marketing_priority', 'k12_pop', 'income']
                ]
                st.dataframe(top_marketing, width='stretch')

        with col3:
            st.write("**Top 10 Block Groups by HPFI**")
            if 'hpfi' in demographics_filtered.columns:
                top_hpfi = demographics_filtered.nlargest(10, 'hpfi')[
                    ['block_group_id', 'hpfi', 'income', 'k12_pop']
                ].copy()
                top_hpfi['hpfi'] = top_hpfi['hpfi'].map(lambda v: f"{v:.2f}")
                top_hpfi['income'] = top_hpfi['income'].map(lambda v: f"${v:,.0f}")
                st.dataframe(top_hpfi, width='stretch')
        
        # Export option
        st.subheader("📥 Export Data")
        if st.button("Download Filtered Data as CSV"):
            csv = demographics_filtered.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"philadelphia_block_groups_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("🔍 No areas match your current filter settings. Try broadening your criteria to see more opportunities.")
        if highlight_hpfi:
            st.info("💡 HPFI Focus Mode is active. Disable 'Focus on High-Potential Family Index' in the sidebar to see all areas.")
        
        # Show available ranges with positive framing
        if not demographics.empty:
            st.info("**📊 Data Available Across Philadelphia:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"• Income Range: ${demographics['income'].min():,.0f} - ${demographics['income'].max():,.0f}")
                st.write(f"• Student Population Range: {demographics['k12_pop'].min():.0f} - {demographics['k12_pop'].max():.0f}")
            with col2:
                st.write(f"• Total Areas Available: {len(demographics)}")
                st.write(f"• Geographic Coverage: Up to 35 km from CCA campuses")

if __name__ == "__main__":
    main()
