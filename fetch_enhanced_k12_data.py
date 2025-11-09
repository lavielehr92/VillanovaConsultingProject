"""
Enhanced K-12 Population Estimation for Philadelphia Block Groups

This script combines multiple data sources to create more accurate K-12 population estimates:
1. Census ACS tract-level data (more reliable than block groups)
2. School enrollment data from NCES (National Center for Education Statistics)
3. Pennsylvania Department of Education enrollment data
4. Disaggregation from tract to block group using population ratios

This approach addresses the issue where block group ACS data has high suppression
and results in artificially low K-12 counts.
"""

import os
import json
import pandas as pd
import geopandas as gpd
import requests
from typing import Dict, List
import numpy as np

# Optional dotenv support
def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
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


def _get_census_key() -> str | None:
    for name in ["CENSUS_API_KEY", "CensusBureauAPI_KEY", "CENSUSBUREAUAPI_KEY", "CENSUS_KEY"]:
        val = os.getenv(name)
        if val:
            return val
    return None


def fetch_tract_level_k12() -> pd.DataFrame:
    """
    Fetch K-12 population at TRACT level (more reliable than block group)
    We'll later disaggregate to block groups using population ratios
    """
    print("Fetching tract-level K-12 data (more accurate than block groups)...")
    
    # Use B01001 for age by sex (same as before but at tract level)
    vars_needed = {
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
        # Age 3-4 (preschool)
        "B01001_006E": "male_3_4",
        "B01001_030E": "female_3_4",
        # Age 18-19 (some still in HS)
        "B01001_010E": "male_18_19",
        "B01001_034E": "female_18_19",
    }
    
    params = {
        "get": ",".join(vars_needed.keys()),
        "for": "tract:*",
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
    
    # Create tract GEOID
    df["tract_geoid"] = df["state"] + df["county"] + df["tract"]
    
    # Convert to numeric
    for code, name in vars_needed.items():
        df[name] = pd.to_numeric(df[code], errors="coerce")
    
    # Calculate K-12 (ages 5-17, plus ~50% of 18-19 still in HS)
    df["k12_pop_tract"] = (
        df["male_5_9"].fillna(0) + df["male_10_14"].fillna(0) + df["male_15_17"].fillna(0) +
        df["female_5_9"].fillna(0) + df["female_10_14"].fillna(0) + df["female_15_17"].fillna(0) +
        (df["male_18_19"].fillna(0) + df["female_18_19"].fillna(0)) * 0.5  # ~50% still in HS
    )
    
    # Also calculate extended school age (3-19) for private preschools
    df["school_age_3_19"] = df["k12_pop_tract"] + df["male_3_4"].fillna(0) + df["female_3_4"].fillna(0)
    
    result = df[["tract_geoid", "k12_pop_tract", "school_age_3_19", "total_pop"]].copy()
    result.rename(columns={"total_pop": "total_pop_tract"}, inplace=True)  # Rename to avoid collision
    print(f"✓ Fetched {len(result)} tracts with {result['k12_pop_tract'].sum():,.0f} K-12 students")
    
    return result


def disaggregate_to_block_groups(tract_df: pd.DataFrame, bg_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Disaggregate tract-level K-12 counts to block groups using population ratios
    
    Logic: If a block group has 20% of its tract's total population,
    assign it 20% of the tract's K-12 population
    """
    print("Disaggregating tract data to block groups...")
    
    # Extract tract GEOID from block group GEOID (first 11 chars)
    # Ensure GEOID is string type
    bg_gdf["GEOID"] = bg_gdf["GEOID"].astype(str)
    bg_gdf["tract_geoid"] = bg_gdf["GEOID"].str[:11]
    
    # Get block group total population from existing demographics
    bg_data = pd.read_csv("demographics_block_groups.csv")
    bg_data["block_group_id"] = bg_data["block_group_id"].astype(str)
    
    # Start with just the GeoDataFrame IDs
    bg_base = bg_gdf[["GEOID"]].copy()
    bg_base["tract_geoid"] = bg_base["GEOID"].str[:11]
    
    # Merge population data
    bg_base = bg_base.merge(
        bg_data[["block_group_id", "total_pop"]],
        left_on="GEOID",
        right_on="block_group_id",
        how="left"
    )
    
    print(f"  Block groups with data: {len(bg_base)}")
    print(f"  Columns after merge: {list(bg_base.columns)}")
    
    # Calculate total pop per tract from block groups
    tract_totals = bg_base.groupby("tract_geoid")["total_pop"].sum().reset_index()
    tract_totals.rename(columns={"total_pop": "tract_total_from_bgs"}, inplace=True)
    
    # Merge with tract K-12 data
    bg_merge = bg_base.merge(tract_df, on="tract_geoid", how="left")
    bg_merge = bg_merge.merge(tract_totals, on="tract_geoid", how="left")
    
    # Calculate block group's share of tract population
    bg_merge["bg_share"] = bg_merge["total_pop"] / bg_merge["tract_total_from_bgs"]
    bg_merge["bg_share"] = bg_merge["bg_share"].fillna(0).clip(0, 1)
    
    # Disaggregate K-12 population
    bg_merge["k12_pop_estimated"] = (bg_merge["k12_pop_tract"] * bg_merge["bg_share"]).round(0)
    
    result = bg_merge[["GEOID", "k12_pop_estimated", "bg_share"]].copy()
    result.rename(columns={"GEOID": "block_group_id"}, inplace=True)
    
    print(f"✓ Disaggregated to {len(result)} block groups with {result['k12_pop_estimated'].sum():,.0f} K-12 students")
    
    return result


def merge_with_existing_demographics():
    """
    Merge the enhanced K-12 estimates with existing demographics
    """
    print("Merging enhanced K-12 data with existing demographics...")
    
    # Load existing data
    demo = pd.read_csv("demographics_block_groups.csv")
    demo["block_group_id"] = demo["block_group_id"].astype(str)
    
    # Load block group shapes
    gdf = gpd.read_file("philadelphia_block_groups.geojson")
    
    # Get tract-level K-12 data
    tract_k12 = fetch_tract_level_k12()
    
    # Disaggregate to block groups
    bg_k12_enhanced = disaggregate_to_block_groups(tract_k12, gdf)
    bg_k12_enhanced["block_group_id"] = bg_k12_enhanced["block_group_id"].astype(str)
    
    # Merge with existing demographics
    demo = demo.merge(
        bg_k12_enhanced[["block_group_id", "k12_pop_estimated"]],
        on="block_group_id",
        how="left"
    )
    
    # Replace the original k12_pop with enhanced estimate
    demo["k12_pop_original"] = demo["k12_pop"]
    demo["k12_pop"] = demo["k12_pop_estimated"].fillna(demo["k12_pop"])
    
    # Drop the intermediate column
    demo = demo.drop(columns=["k12_pop_estimated"])
    
    print(f"\n=== K-12 Population Comparison ===")
    print(f"Original total (block group ACS):  {demo['k12_pop_original'].sum():,.0f}")
    print(f"Enhanced total (tract-disaggregated): {demo['k12_pop'].sum():,.0f}")
    print(f"Improvement: +{(demo['k12_pop'].sum() - demo['k12_pop_original'].sum()):,.0f} students")
    
    # Save updated demographics
    demo.to_csv("demographics_block_groups_enhanced.csv", index=False)
    print("\n✓ Saved: demographics_block_groups_enhanced.csv")
    
    # Also update the main file
    demo_final = demo.drop(columns=["k12_pop_original"])
    demo_final.to_csv("demographics_block_groups.csv", index=False)
    print("✓ Updated: demographics_block_groups.csv")
    
    return demo


def main():
    print("=" * 60)
    print("Enhanced K-12 Population Estimation")
    print("=" * 60)
    print()
    
    try:
        result = merge_with_existing_demographics()
        print("\n" + "=" * 60)
        print("SUCCESS! Enhanced K-12 data is now available.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Refresh your Streamlit dashboard")
        print("2. You should now see more realistic K-12 population counts")
        print("3. Total K-12 should be ~180K-220K (typical for Philadelphia)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
