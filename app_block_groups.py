import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler

# --- OPTIONAL IMPORTS ---
try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    from educational_desert_index_bg import haversine_km
    from scripts.utils.data_quality import compute_legitimate_flag as compute_legitimate_flag_module
    from competition_ingest import load_competition_schools
    from school_ingest import load_census_schools
except ImportError:
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371
        dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    def compute_legitimate_flag_module(df):
        df['is_legit'] = True
        return df
        
    def load_competition_schools(**kwargs): return pd.DataFrame()
    def load_census_schools(**kwargs): return pd.DataFrame()

# --- CONFIG ---
st.set_page_config(page_title="CCA Growth Explorer", layout="wide", initial_sidebar_state="expanded")

CCA_CAMPUSES = pd.DataFrame({
    'name': ['CCA Main Campus (58th St)', 'CCA Baltimore Ave Campus'],
    'lat': [39.9386, 39.9508],
    'lon': [-75.2312, -75.2085],
    'address': ['1939 S. 58th St', '4109 Baltimore Ave']
})

# --- SEPTA TRANSIT DATA ---
# Market-Frankford Line stations (key stations for CCA access)
MFL_STATIONS = [
    {"name": "69th St Terminal", "lat": 39.9630, "lon": -75.2588},
    {"name": "63rd St", "lat": 39.9621, "lon": -75.2495},
    {"name": "60th St", "lat": 39.9612, "lon": -75.2415},  # Closest to CCA!
    {"name": "56th St", "lat": 39.9599, "lon": -75.2329},
    {"name": "52nd St", "lat": 39.9596, "lon": -75.2231},
    {"name": "46th St", "lat": 39.9591, "lon": -75.2106},
    {"name": "40th St", "lat": 39.9581, "lon": -75.1999},
    {"name": "34th St", "lat": 39.9567, "lon": -75.1872},
    {"name": "30th St", "lat": 39.9557, "lon": -75.1827},
    {"name": "15th St", "lat": 39.9523, "lon": -75.1654},
    {"name": "City Hall", "lat": 39.9523, "lon": -75.1637},
    {"name": "5th St", "lat": 39.9517, "lon": -75.1500},
    {"name": "2nd St", "lat": 39.9494, "lon": -75.1410},
    {"name": "Spring Garden", "lat": 39.9595, "lon": -75.1491},
    {"name": "Girard", "lat": 39.9681, "lon": -75.1568},
    {"name": "Berks", "lat": 39.9776, "lon": -75.1431},
    {"name": "Erie-Torresdale", "lat": 40.0011, "lon": -75.1254},
    {"name": "Frankford Terminal", "lat": 40.0231, "lon": -75.0809},
]

# Broad Street Line stations (North-South)
BSL_STATIONS = [
    {"name": "Fern Rock", "lat": 40.0457, "lon": -75.1256},
    {"name": "Olney", "lat": 40.0341, "lon": -75.1212},
    {"name": "Logan", "lat": 40.0217, "lon": -75.1456},
    {"name": "Erie", "lat": 40.0086, "lon": -75.1562},
    {"name": "North Phila", "lat": 39.9913, "lon": -75.1561},
    {"name": "Temple U", "lat": 39.9816, "lon": -75.1498},
    {"name": "Cecil B Moore", "lat": 39.9789, "lon": -75.1589},
    {"name": "City Hall", "lat": 39.9523, "lon": -75.1637},
    {"name": "Walnut-Locust", "lat": 39.9473, "lon": -75.1636},
    {"name": "Ellsworth", "lat": 39.9383, "lon": -75.1664},
    {"name": "Tasker-Morris", "lat": 39.9294, "lon": -75.1679},
    {"name": "Snyder", "lat": 39.9218, "lon": -75.1709},
    {"name": "Oregon", "lat": 39.9156, "lon": -75.1715},
    {"name": "AT&T Station (Sports Complex)", "lat": 39.9065, "lon": -75.1720},
]

# Regional Rail stations with connections to Center City
REGIONAL_RAIL_STATIONS = [
    {"name": "30th St Station", "lat": 39.9557, "lon": -75.1827},
    {"name": "Suburban Station", "lat": 39.9541, "lon": -75.1676},
    {"name": "Jefferson Station", "lat": 39.9527, "lon": -75.1582},
    {"name": "Temple U", "lat": 39.9816, "lon": -75.1498},
    {"name": "North Broad", "lat": 39.9697, "lon": -75.1573},
    {"name": "Wayne Junction", "lat": 40.0236, "lon": -75.1595},
    {"name": "Fern Rock", "lat": 40.0457, "lon": -75.1256},
    {"name": "Cheltenham", "lat": 40.0612, "lon": -75.1456},
    {"name": "Elkins Park", "lat": 40.0712, "lon": -75.1266},
    {"name": "Jenkintown-Wyncote", "lat": 40.0956, "lon": -75.1370},
    {"name": "Glenside", "lat": 40.1045, "lon": -75.1531},
    {"name": "Ardmore", "lat": 40.0089, "lon": -75.2911},
    {"name": "Bryn Mawr", "lat": 40.0192, "lon": -75.3063},
    {"name": "Overbrook", "lat": 39.9873, "lon": -75.2515},
    {"name": "Merion", "lat": 39.9989, "lon": -75.2631},
]

# Key bus routes serving CCA area
# Route 42 (Spruce/Pine) - Direct to CCA
# Route 34 Trolley - Connects to 60th St
CCA_BUS_CORRIDORS = [
    {"name": "Route 42 - Spruce St", "lat_range": (39.935, 39.945), "lon_range": (-75.25, -75.15)},
]

# --- DATA LOADING ---
@st.cache_data(ttl=3600)
def load_core_data():
    try:
        # Try metro-wide data first, fall back to Philadelphia-only
        if os.path.exists('metro_block_groups.geojson'):
            gdf = gpd.read_file('metro_block_groups.geojson')
            demos = pd.read_csv('demographics_metro.csv')
        else:
            gdf = gpd.read_file('philadelphia_block_groups.geojson')
            demos = pd.read_csv('demographics_block_groups.csv')
        
        if 'GEOID' in gdf.columns: gdf['GEOID'] = gdf['GEOID'].astype(str)
        if 'block_group_id' in demos.columns: demos['block_group_id'] = demos['block_group_id'].astype(str)
        
        cols = ['income', 'poverty_rate', 'total_pop', 'k12_pop', '%Christian']
        for c in cols:
            if c in demos.columns:
                demos[c] = pd.to_numeric(demos[c], errors='coerce')
        
        return gdf, demos
    except Exception as e:
        st.error(f"Data Error: {e}")
        return gpd.GeoDataFrame(), pd.DataFrame()

@st.cache_data
def load_aux_layers():
    comp = load_competition_schools()
    public = load_census_schools()
    try:
        students = pd.read_csv('current_students_anonymized.csv')
    except:
        students = pd.DataFrame()
    return comp, public, students


# --- SAFETY, COMMUTE, AND TRANSIT SCORING ---

def calc_safety_score(lat, lon):
    """
    Estimate neighborhood safety based on location.
    Returns 0-1 score (1 = safest).
    Based on Philadelphia crime patterns by neighborhood.
    """
    if pd.isna(lat) or pd.isna(lon):
        return 0.5
    
    # === HIGHER CRIME AREAS (score 0.2-0.4) ===
    # North Philadelphia - Kensington/Badlands
    if 39.98 <= lat <= 40.02 and -75.15 <= lon <= -75.10:
        return 0.25
    # North Philadelphia - general
    if 39.97 <= lat <= 40.03 and -75.18 <= lon <= -75.12:
        return 0.35
    # Southwest Philadelphia
    if 39.91 <= lat <= 39.94 and -75.24 <= lon <= -75.20:
        return 0.40
    # West Philadelphia - deeper areas
    if 39.95 <= lat <= 39.97 and -75.26 <= lon <= -75.22:
        return 0.45
    
    # === MODERATE AREAS (score 0.5-0.65) ===
    # West Philadelphia - near CCA
    if 39.93 <= lat <= 39.96 and -75.24 <= lon <= -75.20:
        return 0.55
    # South Philadelphia
    if 39.91 <= lat <= 39.94 and -75.18 <= lon <= -75.14:
        return 0.55
    # Germantown
    if 40.03 <= lat <= 40.07 and -75.18 <= lon <= -75.14:
        return 0.50
    # Northeast Philadelphia - lower
    if 40.02 <= lat <= 40.06 and -75.10 <= lon <= -75.02:
        return 0.60
    
    # === SAFER AREAS (score 0.7-0.85) ===
    # Center City
    if 39.94 <= lat <= 39.96 and -75.18 <= lon <= -75.15:
        return 0.75
    # University City
    if 39.94 <= lat <= 39.96 and -75.21 <= lon <= -75.18:
        return 0.70
    # Chestnut Hill / Mt Airy
    if 40.05 <= lat <= 40.10 and -75.22 <= lon <= -75.18:
        return 0.80
    # Manayunk / Roxborough
    if 40.02 <= lat <= 40.06 and -75.24 <= lon <= -75.20:
        return 0.75
    # Far Northeast
    if 40.05 <= lat <= 40.12 and -75.05 <= lon <= -74.98:
        return 0.75
    
    # === PA SUBURBS (score 0.8-0.95) ===
    # Lower Merion / Main Line
    if 39.95 <= lat <= 40.08 and -75.35 <= lon <= -75.25:
        return 0.90
    # Delaware County suburbs
    if 39.88 <= lat <= 39.95 and -75.35 <= lon <= -75.25:
        return 0.85
    
    # === NJ SUBURBS (score 0.8-0.95) ===
    if lon > -75.13:
        # Camden City - higher crime
        if 39.92 <= lat <= 39.97 and -75.13 <= lon <= -75.08:
            return 0.35
        # NJ suburbs generally safe
        return 0.85
    
    # Default Philadelphia
    return 0.55


def calc_commute_score(lat, lon, dist_km):
    """
    Calculate commute score and estimated drive time to CCA.
    Returns (score 0-1, estimated_minutes)
    Higher score = better (shorter commute).
    All commutes assumed to be driving.
    
    Realistic estimates based on Philadelphia traffic:
    - Urban core: 4-5 min/km (stop lights, traffic, parking)
    - Suburban to urban: 3-4 min/km + 5-10 min base
    - NJ: Add 10-15 min for bridge crossing
    """
    if pd.isna(lat) or pd.isna(lon):
        return 0.5, 30
    
    # Base time for getting in/out of car, parking, etc.
    base_time = 5
    
    # Check location type
    is_urban_core = (39.93 <= lat <= 39.97 and -75.22 <= lon <= -75.15)  # West Philly/University City
    is_urban = (39.90 <= lat <= 40.05 and -75.25 <= lon <= -75.10)  # Greater Philadelphia
    is_nj = lon > -75.10
    
    if is_urban_core:
        # Very close to CCA but city traffic - 5 min/km
        minutes = base_time + dist_km * 5
    elif is_urban:
        # Urban Philadelphia - lots of lights, traffic - 4 min/km + base
        minutes = base_time + 5 + dist_km * 4
    elif is_nj:
        # NJ - bridge crossing adds significant time
        bridge_time = 12  # Ben Franklin or Walt Whitman bridge delay
        minutes = base_time + bridge_time + dist_km * 3
    else:
        # PA suburbs - faster roads but still need to get into city
        # Upper Darby to University City example: ~5km, should be ~25 min
        minutes = base_time + 8 + dist_km * 3.5
    
    # Cap at reasonable values
    minutes = max(8, min(minutes, 65))  # At least 8 min drive anywhere
    
    # Score: 0 at 50+ min, 1.0 at 8 min
    score = max(0, 1 - (minutes - 8) / 42)
    
    return round(score, 2), int(minutes)


def calc_transit_score(lat, lon):
    """
    Calculate transit accessibility score for getting to CCA.
    Returns (score 0-1, transit_type, can_student_commute)
    
    CCA Main Campus is near 60th St MFL station (~0.5 mile walk).
    """
    if pd.isna(lat) or pd.isna(lon):
        return 0.3, "None", False
    
    # Check distance to nearest MFL station
    min_mfl_dist = float('inf')
    nearest_mfl = None
    for station in MFL_STATIONS:
        dist = haversine_km(lat, lon, station['lat'], station['lon'])
        if dist < min_mfl_dist:
            min_mfl_dist = dist
            nearest_mfl = station['name']
    
    # Check distance to nearest BSL station
    min_bsl_dist = float('inf')
    for station in BSL_STATIONS:
        dist = haversine_km(lat, lon, station['lat'], station['lon'])
        if dist < min_bsl_dist:
            min_bsl_dist = dist
    
    # Check distance to nearest Regional Rail
    min_rr_dist = float('inf')
    for station in REGIONAL_RAIL_STATIONS:
        dist = haversine_km(lat, lon, station['lat'], station['lon'])
        if dist < min_rr_dist:
            min_rr_dist = dist
    
    # Distance from this location to CCA
    dist_to_cca = min([haversine_km(lat, lon, c['lat'], c['lon']) 
                       for _, c in CCA_CAMPUSES.iterrows()])
    
    # Scoring logic
    # Best: Near MFL (direct line to 60th St near CCA)
    # Good: Near BSL (transfer to MFL at City Hall)
    # OK: Near Regional Rail (transfer downtown)
    # Poor: No rail access
    
    if dist_to_cca <= 1.6:  # Within 1 mile of CCA
        return 1.0, "Walk to CCA", True
    
    if min_mfl_dist <= 0.8:  # Within ~0.5 mile of MFL
        return 0.9, f"MFL ({nearest_mfl})", True
    
    if min_mfl_dist <= 1.5:  # Within ~1 mile of MFL
        return 0.75, f"MFL (walk)", True
    
    if min_bsl_dist <= 0.8:  # Near BSL (1 transfer to CCA)
        return 0.65, "BSL + Transfer", True
    
    if min_rr_dist <= 1.0:  # Near Regional Rail
        return 0.55, "Regional Rail", True
    
    if min_bsl_dist <= 1.5 or min_rr_dist <= 1.5:
        return 0.4, "Transit (far)", False
    
    # NJ - would need Patco + transfer
    if lon > -75.10:
        return 0.3, "PATCO + Transfers", False
    
    return 0.2, "Drive Only", False

# --- CALCULATIONS ---
def enrich_data(demos, weights, schools_df=None):
    df = demos.copy()
    
    # EDI (Educational Desert Index) - measures LACK of QUALITY school access
    # 
    # KEY INSIGHT: Access works DIFFERENTLY in different areas:
    # 
    # SUBURBAN NJ/PA: Guaranteed public school access
    #   - Every resident has a SEAT in their district school
    #   - No lottery, no waitlist, no rejection
    #   - If district quality >= 0.75, area is WELL-SERVED (low EDI)
    #
    # PHILADELPHIA CITY: Competitive/lottery-based access
    #   - Public schools exist but quality is low (avg 0.45)
    #   - Good options (charters) have lotteries - 16K+ on waitlists
    #   - Having a school nearby ≠ having ACCESS to quality education
    #   - Must calculate supply/demand ratio for TRUE access
    #
    # Higher EDI = educational desert (no guaranteed quality access)
    # Lower EDI = well-served (guaranteed access to quality school)
    
    def get_guaranteed_public_quality(lat, lon):
        """
        Check if location has GUARANTEED access to a quality public school.
        Returns the quality rating if guaranteed access exists, else None.
        
        In NJ/PA suburbs, residents are guaranteed a seat in their district school.
        In Philadelphia, public schools are low quality and good charters are lottery-based.
        """
        if pd.isna(lat) or pd.isna(lon):
            return None
        
        # === NEW JERSEY (lon > -75.13 generally) ===
        if lon > -75.13:
            # Haddonfield (A+ district) - guaranteed 0.95
            if 39.88 <= lat <= 39.91 and -75.04 <= lon <= -75.02:
                return 0.95
            
            # Moorestown (A+ district) - guaranteed 0.95
            if 39.96 <= lat <= 40.00 and -74.97 <= lon <= -74.92:
                return 0.95
            
            # Cherry Hill (A district) - guaranteed 0.85
            if 39.87 <= lat <= 39.93 and -75.02 <= lon <= -74.94:
                return 0.85
            
            # Voorhees (A- district) - guaranteed 0.80
            if 39.82 <= lat <= 39.87 and -74.98 <= lon <= -74.90:
                return 0.80
            
            # Medford/Lenape (A- district) - guaranteed 0.80
            if 39.80 <= lat <= 39.92 and -74.93 <= lon <= -74.80:
                return 0.80
            
            # Collingswood (B+ district) - guaranteed 0.75
            if 39.90 <= lat <= 39.93 and -75.08 <= lon <= -75.05:
                return 0.75
            
            # Washington Township, Gloucester (B+ district) - guaranteed 0.75
            if 39.74 <= lat <= 39.78 and -75.08 <= lon <= -75.04:
                return 0.75
            
            # West Deptford (B+ district) - guaranteed 0.75
            if 39.82 <= lat <= 39.86 and -75.15 <= lon <= -75.10:
                return 0.75
            
            # Deptford Township (B district) - guaranteed 0.70
            if 39.78 <= lat <= 39.83 and -75.12 <= lon <= -75.05:
                return 0.70
            
            # Mantua Township (B district) - guaranteed 0.70
            if 39.78 <= lat <= 39.82 and -75.18 <= lon <= -75.12:
                return 0.70
            
            # Woodbury (B district) - guaranteed 0.70
            if 39.82 <= lat <= 39.85 and -75.17 <= lon <= -75.13:
                return 0.70
            
            # Paulsboro/low-income Gloucester areas - lower quality
            if 39.82 <= lat <= 39.86 and -75.25 <= lon <= -75.18:
                return 0.55
            
            # Camden City - NO quality guaranteed (public schools ~0.40)
            if 39.92 <= lat <= 39.97 and -75.13 <= lon <= -75.08:
                return None  # Must compete for charters
            
            # Other NJ suburban areas - assume decent public (B level)
            if lon > -75.10:
                return 0.70
        
        # === PENNSYLVANIA SUBURBS (west of Philly) ===
        # Lower Merion (A+ district) - guaranteed 0.95
        if 39.95 <= lat <= 40.08 and -75.35 <= lon <= -75.20:
            return 0.95
        
        # Radnor (A+ district) - guaranteed 0.95
        if 39.96 <= lat <= 40.05 and -75.42 <= lon <= -75.33:
            return 0.95
        
        # Wallingford-Swarthmore (A+ district) - guaranteed 0.95
        if 39.88 <= lat <= 39.92 and -75.38 <= lon <= -75.33:
            return 0.95
        
        # Rose Tree Media (A district) - guaranteed 0.85
        if 39.90 <= lat <= 39.94 and -75.42 <= lon <= -75.37:
            return 0.85
        
        # Haverford Township (A district) - guaranteed 0.85
        if 39.94 <= lat <= 40.00 and -75.32 <= lon <= -75.26:
            return 0.85
        
        # Marple Newtown (A- district) - guaranteed 0.80
        if 39.92 <= lat <= 39.97 and -75.38 <= lon <= -75.33:
            return 0.80
        
        # Springfield Delco (A- district) - guaranteed 0.80
        if 39.90 <= lat <= 39.95 and -75.37 <= lon <= -75.30:
            return 0.80
        
        # Upper Darby (B district) - guaranteed but lower quality 0.65
        if 39.94 <= lat <= 39.98 and -75.30 <= lon <= -75.24:
            return 0.65
        
        # Ridley (B district) - guaranteed 0.70
        if 39.86 <= lat <= 39.90 and -75.35 <= lon <= -75.30:
            return 0.70
        
        # Chester-Upland (struggling district) - guaranteed but low quality
        if 39.84 <= lat <= 39.87 and -75.38 <= lon <= -75.33:
            return 0.45
        
        # General Delaware County suburbs not otherwise specified
        if 39.85 <= lat <= 39.98 and -75.45 <= lon <= -75.30:
            return 0.75  # Assume decent suburban public
        
        # === PHILADELPHIA CITY ===
        # NO guaranteed quality access - public schools avg 0.45
        # Good schools are charter-based (lottery) with 16K+ waitlists
        if 39.87 <= lat <= 40.14 and -75.28 <= lon <= -75.05:
            return None  # Must use competitive model
        
        # Default: unknown area, use competitive model
        return None
    
    def calc_edi(row, schools, all_blocks):
        if schools is None or schools.empty:
            return 50
        
        lat, lon = row['lat'], row['lon']
        k12_local = row.get('k12_pop', 100)
        
        if pd.isna(lat) or pd.isna(lon):
            return 50
        
        # FIRST: Check for guaranteed public school access
        guaranteed_quality = get_guaranteed_public_quality(lat, lon)
        
        if guaranteed_quality is not None:
            # Area has GUARANTEED access to a public school of this quality
            # Convert quality to EDI: higher quality = lower EDI
            # Quality 0.95 → EDI ~5 (excellent guaranteed access)
            # Quality 0.75 → EDI ~25 (good guaranteed access)
            # Quality 0.70 → EDI ~30 (decent guaranteed access)
            edi = max(0, (1 - guaranteed_quality) * 100)
            return edi
        
        # NO guaranteed quality access - use COMPETITIVE model
        # This applies to Philadelphia and Camden where good schools are lottery-based
        
        # Estimate capacity by school type
        CAPACITY_BY_TYPE = {
            'Public': 600,           # Large but LOW quality in Philly
            'High School': 800,
            'Middle School': 500,
            'Elementary School': 400,
            'Charter': 350,          # Good but LOTTERY-BASED (16K waitlist)
            'Catholic': 300,
            'Christian': 250,
            'Private': 250,
            'K-12 School': 400
        }
        
        # Calculate competitive access to QUALITY schools only
        # In Philly, having a nearby public school doesn't help if it's 0.45 quality
        school_data = []
        for _, school in schools.iterrows():
            if pd.notna(school['lat']) and pd.notna(school['lon']):
                dist = haversine_km(lat, lon, school['lat'], school['lon'])
                quality = school.get('quality_rating', 0.5)
                school_type = school.get('type', 'K-12 School')
                capacity = CAPACITY_BY_TYPE.get(school_type, 500)
                
                # Only count schools with quality >= 0.6 as viable options
                if quality >= 0.6:
                    # Reduce effective capacity for charter/private (lottery/tuition barriers)
                    if school_type in ['Charter', 'Catholic', 'Christian', 'Private']:
                        effective_cap = capacity * 0.3  # Only ~30% chance of getting in
                    else:
                        effective_cap = capacity * quality  # Public weighted by quality
                    
                    school_data.append({
                        'dist': dist,
                        'quality': quality,
                        'capacity': capacity,
                        'effective_seats': effective_cap
                    })
        
        if not school_data:
            return 95  # No quality options = severe desert
        
        school_data.sort(key=lambda x: x['dist'])
        
        # Calculate supply/demand for QUALITY education
        effective_seats_5km = sum(s['effective_seats'] for s in school_data if s['dist'] <= 5)
        
        # In urban areas, many more students compete for same schools
        # Philadelphia has ~200K K-12 students, very high density
        local_demand = max(k12_local * 40, 800)
        
        # Supply/Demand ratio
        supply_demand = effective_seats_5km / local_demand
        
        # Convert to EDI score (50-100 range for competitive areas)
        # Ratio 0.5+ = EDI 50 (some access), Ratio 0.1 = EDI 90 (severe shortage)
        edi = max(50, min(95, 100 - (supply_demand * 100)))
        
        return edi
    
    if 'EDI' not in df.columns:
        df['EDI'] = df.apply(lambda r: calc_edi(r, schools_df, df), axis=1)
    
    # HPFI Calculation
    scaler = MinMaxScaler()
    for c in ['income', 'k12_pop', 'poverty_rate', '%Christian']:
        if c not in df.columns: df[c] = 0
    
    df['norm_income'] = scaler.fit_transform(df[['income']].fillna(0))
    df['norm_k12'] = scaler.fit_transform(df[['k12_pop']].fillna(0))
    df['norm_inv_poverty'] = 1 - scaler.fit_transform(df[['poverty_rate']].fillna(0))
    df['norm_christian'] = scaler.fit_transform(df[['%Christian']].fillna(0))
    
    df['min_dist_km'] = df.apply(lambda r: min([haversine_km(r['lat'], r['lon'], c['lat'], c['lon']) for _,c in CCA_CAMPUSES.iterrows()]), axis=1)
    df['norm_prox'] = np.exp(-df['min_dist_km'] / 8.0)
    
    # Safety Score
    df['safety_score'] = df.apply(lambda r: calc_safety_score(r['lat'], r['lon']), axis=1)
    
    # Commute scoring (driving only) - returns (score, minutes)
    commute_results = df.apply(lambda r: calc_commute_score(r['lat'], r['lon'], r['min_dist_km']), axis=1)
    df['commute_score'] = commute_results.apply(lambda x: x[0])
    df['commute_minutes'] = commute_results.apply(lambda x: x[1])
    
    # HPFI Calculation - now includes safety weight
    df['hpfi'] = (
        weights['income'] * df['norm_income'] +
        weights['poverty'] * df['norm_inv_poverty'] +
        weights['proximity'] * df['norm_prox'] +
        weights['christian'] * df['norm_christian'] +
        weights['k12'] * df['norm_k12'] +
        weights.get('safety', 0) * df['safety_score']
    ).clip(0, 1)
    
    # Priority Zones
    hpfi_75 = df['hpfi'].quantile(0.75)
    edi_75 = df['EDI'].quantile(0.75)
    
    def get_zone(r):
        high_h = r['hpfi'] >= hpfi_75
        high_e = r['EDI'] >= edi_75
        if high_h and high_e: return "High Priority"
        if high_h: return "Strong Potential"
        if high_e: return "Underserved"
        return "Low Priority"
    
    df['zone'] = df.apply(get_zone, axis=1)
    return df

# --- MAIN APP ---
def main():
    st.title("CCA Growth Explorer")
    
    # Quick action buttons at top
    col_title1, col_title2, col_title3 = st.columns([2, 1, 1])
    with col_title2:
        if st.button("Reset Filters", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    with col_title3:
        show_help = st.button("Help", use_container_width=True)
    
    if show_help:
        st.info("""
        **Quick Start:** Adjust filters in the sidebar to find high-potential neighborhoods. 
        Look for "High Priority" zones (green) - these are affluent areas with limited school options.
        """)
    
    # Usage Instructions (collapsed by default)
    with st.expander("How to Use This Dashboard", expanded=False):
        st.markdown("""
        **CCA Growth Explorer** helps identify high-potential neighborhoods for Cornerstone Christian Academy enrollment growth.
        
        **Getting Started:**
        1. Use the **Filters** in the left sidebar to narrow down areas of interest
        2. Adjust the **distance slider** to expand or contract the search radius from CCA campuses
        3. Set **income range** to focus on target demographics
        4. Use **Min K-12 Population** to exclude areas with few school-age children
        
        **Understanding the Scores:**
        - **HPFI (High Potential Family Index):** 0-1 score combining income, proximity, K-12 population, and other factors. Higher = more potential.
        - **EDI (Educational Desert Index):** Measures how underserved an area is. Higher = greater need for quality education options.
        
        **Priority Zones:**
        - **High Priority:** High HPFI + High EDI — Best opportunities (affluent AND underserved)
        - **Strong Potential:** High HPFI — Good family demographics
        - **Underserved:** High EDI — Mission-aligned but may need financial aid support
        - **Low Priority:** Lower scores on both metrics
        
        **Using the Map:**
        - Color the map by different metrics using the dropdown
        - Hover over areas to see detailed data
        - Gold circles mark CCA campus locations
        - Toggle competitor schools on/off with the checkbox
        
        **Exporting Data:**
        - Switch to the **Data** tab to view the full table
        - Click **Download CSV** to export filtered results for further analysis
        """)
    
    # Load Data
    with st.spinner("Loading data..."):
        gdf, raw_demos = load_core_data()
        comp_schools, pub_schools, students_df = load_aux_layers()
    
    if raw_demos.empty:
        st.error("No demographic data found.")
        st.stop()
    
    # Clean income data - Census uses negative values for missing data
    raw_demos['income'] = raw_demos['income'].apply(lambda x: x if x > 0 else np.nan)
    
    # --- SIDEBAR: ALL FILTERS ---
    st.sidebar.header("Filters")
    
    # Reset button in sidebar too
    if st.sidebar.button("Reset All Filters", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    st.sidebar.divider()
    
    # Distance
    radius = st.sidebar.slider("Max Distance from Campus (km)", 1, 35, 15, key="radius")
    
    # Income - filter out missing/invalid values for range calculation
    valid_incomes = raw_demos['income'].dropna()
    min_inc = int(valid_incomes.min()) if len(valid_incomes) > 0 else 0
    max_inc = int(valid_incomes.max()) if len(valid_incomes) > 0 else 250000
    income_range = st.sidebar.slider("Income Range ($)", min_inc, max_inc, (min_inc, max_inc), step=5000, key="income")
    
    # K-12 Population
    min_k12 = st.sidebar.number_input("Min K-12 Population", 0, 1000, 0, key="k12")
    
    # HPFI Threshold
    hpfi_min = st.sidebar.slider("Min HPFI Score", 0.0, 1.0, 0.0, 0.05)
    
    # Commute Filter
    max_commute = st.sidebar.slider("Max Commute (minutes)", 10, 60, 45, 5)
    
    # Data Quality
    hide_sparse = st.sidebar.checkbox("Hide areas with 0 population", value=True)
    
    # Zone Filter
    zone_filter = st.sidebar.multiselect(
        "Priority Zones",
        ["High Priority", "Strong Potential", "Underserved", "Low Priority"],
        default=["High Priority", "Strong Potential", "Underserved", "Low Priority"]
    )
    
    # Weight Adjustments
    with st.sidebar.expander("Score Weights"):
        weights = {
            'income': st.slider("Income", 0.0, 1.0, 0.40, 0.05),
            'poverty': st.slider("Low Poverty", 0.0, 1.0, 0.15, 0.05),
            'proximity': st.slider("Proximity", 0.0, 1.0, 0.10, 0.05),
            'christian': st.slider("Christian %", 0.0, 1.0, 0.10, 0.05),
            'k12': st.slider("K-12 Pop", 0.0, 1.0, 0.10, 0.05),
            'safety': st.slider("Safety", 0.0, 1.0, 0.15, 0.05),
        }
    
    # Load schools with quality ratings for EDI calculation
    if os.path.exists('schools_with_quality.csv'):
        all_schools = pd.read_csv('schools_with_quality.csv')
    else:
        # Fallback to combining without quality ratings
        all_schools = pd.concat([comp_schools, pub_schools], ignore_index=True) if not pub_schools.empty else comp_schools
        all_schools['quality_rating'] = 0.5  # Default rating
    
    # Enrich full dataset
    full_data = enrich_data(raw_demos, weights, all_schools)
    full_data = compute_legitimate_flag_module(full_data)
    
    # Apply Filters
    df = full_data.copy()
    df['dist_check'] = df.apply(lambda r: min([haversine_km(r['lat'], r['lon'], c['lat'], c['lon']) for _,c in CCA_CAMPUSES.iterrows()]), axis=1)
    df = df[df['dist_check'] <= radius]
    df = df[df['income'].between(income_range[0], income_range[1])]
    df = df[df['k12_pop'] >= min_k12]
    df = df[df['hpfi'] >= hpfi_min]
    df = df[df['commute_minutes'] <= max_commute]
    df = df[df['zone'].isin(zone_filter)]
    if hide_sparse:
        df = df[df['total_pop'] > 0]
    df = df[df['is_legit'] == True]
    
    # --- KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Block Groups", len(df))
    c2.metric("K-12 Students", f"{df['k12_pop'].sum():,.0f}")
    c3.metric("Avg HPFI", f"{df['hpfi'].mean():.2f}")
    c4.metric("Avg Income", f"${df['income'].mean():,.0f}")
    
    # --- ACTIONABLE INSIGHTS ---
    st.divider()
    
    # Calculate top opportunities
    if len(df) > 0:
        # Top 5 neighborhoods by HPFI (most likely to convert)
        top_hpfi = df.nlargest(5, 'hpfi')[['block_group_id', 'hpfi', 'income', 'k12_pop', 'zone', 'EDI', 'commute_minutes']]
        
        # High Priority zones (affluent + underserved = sweet spot)
        high_priority = df[df['zone'] == 'High Priority']
        
        # Calculate enrollment potential (rough estimate: 1-2% of K-12 population as potential enrollees)
        estimated_potential = int(df['k12_pop'].sum() * 0.015)  # 1.5% capture rate
        
        # Identify best marketing opportunities by county
        if 'state_fips' in df.columns:
            df['state'] = df['state_fips'].apply(lambda x: 'PA' if x == '42' else 'NJ')
            state_summary = df.groupby('state').agg({
                'k12_pop': 'sum',
                'hpfi': 'mean',
                'income': 'mean'
            }).round(2)
        
        insight_col1, insight_col2 = st.columns([1, 1])
        
        with insight_col1:
            st.subheader("Top 5 Target Neighborhoods")
            st.markdown("*These areas have the highest potential for enrollment growth:*")
            
            for i, (_, row) in enumerate(top_hpfi.iterrows(), 1):
                zone_label = f"({row['zone']})"
                st.markdown(f"""
                **{i}. Block Group {row['block_group_id'][-7:]}** {zone_label}
                - HPFI Score: **{row['hpfi']:.2f}** | Income: **${row['income']:,.0f}**
                - K-12 Students: **{row['k12_pop']:,.0f}** | Commute: **{row['commute_minutes']:.0f} min**
                """)
        
        with insight_col2:
            st.subheader("Enrollment Opportunity Summary")
            
            # Key metrics
            st.metric("Estimated Enrollment Potential", f"{estimated_potential:,} students", 
                     help="Based on 1.5% capture rate of K-12 population in filtered areas")
            
            st.metric("High Priority Areas", f"{len(high_priority):,} neighborhoods",
                     help="Areas with both high income AND limited school options")
            
            if len(high_priority) > 0:
                hp_students = high_priority['k12_pop'].sum()
                st.metric("Students in High Priority Zones", f"{hp_students:,.0f}")
            
            # Quick wins
            st.markdown("---")
            st.markdown("**Quick Win Recommendations:**")
            if len(high_priority) > 0:
                st.success(f"Focus marketing on **{len(high_priority)}** High Priority areas with **{high_priority['k12_pop'].sum():,.0f}** students")
            
            avg_commute = df['commute_minutes'].mean()
            if avg_commute < 30:
                st.success(f"Average commute is **{avg_commute:.0f} min** - emphasize convenient location")
            else:
                st.warning(f"Average commute is **{avg_commute:.0f} min** - consider transportation partnerships")
            
            # EDI insight
            high_edi = df[df['EDI'] >= 60]
            if len(high_edi) > 0:
                st.info(f"**{len(high_edi)}** areas show high educational need (EDI >= 60)")
    
    st.divider()
    
    # --- TABS ---
    tab_map, tab_data, tab_forecast = st.tabs(["Map", "Data", "Forecasting"])
    
    with tab_map:
        gdf_filtered = gdf[gdf['GEOID'].isin(df['block_group_id'])]
        map_df = gdf_filtered.merge(df, left_on='GEOID', right_on='block_group_id', suffixes=('_geo', ''))
        
        color_col = st.selectbox("Color by:", ['hpfi', 'zone', 'EDI', 'income', 'k12_pop', 'safety_score', 'commute_minutes'])
        
        if not map_df.empty:
            # Reverse color scale for commute (lower is better)
            if color_col == 'commute_minutes':
                color_scale = "RdYlGn_r"
            elif color_col == 'EDI':
                color_scale = "RdYlGn_r"  # Higher EDI = worse = red
            elif color_col in ['hpfi', 'income', 'safety_score', 'transit_score']:
                color_scale = "RdYlGn"
            else:
                color_scale = "Viridis"
            
            fig = px.choropleth_mapbox(
                map_df,
                geojson=json.loads(map_df.geometry.to_json()),
                locations=map_df.index,
                color=color_col,
                color_continuous_scale=color_scale,
                mapbox_style="carto-positron",
                center={"lat": 40.0, "lon": -75.15},
                zoom=10,
                opacity=0.6,
                hover_data={
                    'GEOID': True,
                    'income': ':$,.0f',
                    'k12_pop': True,
                    'zone': True,
                    'hpfi': ':.2f',
                    'EDI': ':.0f',
                    'safety_score': ':.2f',
                    'commute_minutes': True
                },
                height=600
            )
            
            fig.add_trace(go.Scattermapbox(
                lat=CCA_CAMPUSES['lat'], lon=CCA_CAMPUSES['lon'],
                mode='markers+text',
                marker=go.scattermapbox.Marker(
                    size=20, 
                    color='gold',
                    opacity=1
                ),
                text=CCA_CAMPUSES['name'],
                textposition='top center',
                textfont=dict(size=11, color='black'),
                name="CCA Campuses",
                hovertext=CCA_CAMPUSES['address']
            ))
            
            # Add a second layer for star effect (smaller dark center)
            fig.add_trace(go.Scattermapbox(
                lat=CCA_CAMPUSES['lat'], lon=CCA_CAMPUSES['lon'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=8, 
                    color='darkgoldenrod',
                    opacity=1
                ),
                name="CCA (center)",
                showlegend=False
            ))
            
            show_competitors = st.checkbox("Show Competitor Schools")
            if show_competitors and not comp_schools.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=comp_schools['lat'], lon=comp_schools['lon'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(size=8, color='red'),
                    text=comp_schools.get('school_name', ''),
                    name="Competitors"
                ))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data matches current filters.")
    
    with tab_data:
        display_cols = ['block_group_id', 'zone', 'hpfi', 'EDI', 'income', 'k12_pop', 'safety_score', 
                        'commute_minutes', 'min_dist_km']
        available_cols = [c for c in display_cols if c in df.columns]
        display_df = df[available_cols].sort_values('hpfi', ascending=False)
        
        st.dataframe(display_df, use_container_width=True, height=500)
        
        csv = display_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "cca_growth_data.csv", "text/csv")
    
    with tab_forecast:
        st.subheader("Enrollment Growth Forecasting")
        st.markdown("*Project future enrollment based on current data and growth assumptions*")
        
        # Forecasting inputs
        fc1, fc2, fc3 = st.columns(3)
        
        with fc1:
            current_enrollment = st.number_input("Current CCA Enrollment", value=500, min_value=0, step=50)
        with fc2:
            capture_rate = st.slider("Target Capture Rate (%)", 0.5, 5.0, 1.5, 0.25)
        with fc3:
            growth_years = st.slider("Forecast Period (Years)", 1, 10, 5)
        
        st.divider()
        
        # Calculate projections
        total_k12_pool = df['k12_pop'].sum()
        high_priority_pool = df[df['zone'] == 'High Priority']['k12_pop'].sum() if len(df[df['zone'] == 'High Priority']) > 0 else 0
        strong_potential_pool = df[df['zone'] == 'Strong Potential']['k12_pop'].sum() if len(df[df['zone'] == 'Strong Potential']) > 0 else 0
        
        # Different capture rates by zone
        hp_capture = capture_rate * 2 / 100  # High Priority has 2x capture rate
        sp_capture = capture_rate * 1.5 / 100  # Strong Potential has 1.5x capture rate
        base_capture = capture_rate / 100
        
        # Year-over-year projections
        years = list(range(growth_years + 1))
        enrollment_projection = [current_enrollment]
        
        # Assume graduated growth (takes time to reach full capture rate)
        for year in range(1, growth_years + 1):
            year_factor = min(year / 3, 1.0)  # Reaches full potential by year 3
            
            new_from_hp = high_priority_pool * hp_capture * year_factor
            new_from_sp = strong_potential_pool * sp_capture * year_factor
            other_pool = total_k12_pool - high_priority_pool - strong_potential_pool
            new_from_other = other_pool * base_capture * year_factor
            
            # Account for attrition (5% per year)
            retained = enrollment_projection[-1] * 0.95
            new_enrollment = retained + new_from_hp + new_from_sp + new_from_other
            enrollment_projection.append(int(new_enrollment))
        
        # Display forecast
        fc_col1, fc_col2 = st.columns([2, 1])
        
        with fc_col1:
            # Create projection chart
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=years,
                y=enrollment_projection,
                mode='lines+markers',
                name='Projected Enrollment',
                line=dict(color='blue', width=3),
                marker=dict(size=10)
            ))
            
            # Add target line
            target = current_enrollment * 2  # Double enrollment as target
            fig_forecast.add_hline(y=target, line_dash="dash", line_color="green", 
                                  annotation_text=f"Target: {target}")
            
            fig_forecast.update_layout(
                title="Enrollment Growth Projection",
                xaxis_title="Years",
                yaxis_title="Total Enrollment",
                height=400
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        with fc_col2:
            st.markdown("### Key Projections")
            
            year_5_proj = enrollment_projection[min(5, growth_years)] if len(enrollment_projection) > 5 else enrollment_projection[-1]
            year_3_proj = enrollment_projection[min(3, growth_years)] if len(enrollment_projection) > 3 else enrollment_projection[-1]
            
            st.metric(f"Year {growth_years} Projection", f"{enrollment_projection[-1]:,}")
            st.metric("Potential Growth", f"+{enrollment_projection[-1] - current_enrollment:,}", 
                     f"{((enrollment_projection[-1] - current_enrollment) / current_enrollment * 100):.1f}%")
            
            st.markdown("---")
            st.markdown("### Addressable Market")
            st.markdown(f"- Total K-12 Pool: **{total_k12_pool:,.0f}**")
            st.markdown(f"- High Priority Pool: **{high_priority_pool:,.0f}**")
            st.markdown(f"- Strong Potential Pool: **{strong_potential_pool:,.0f}**")
        
        st.divider()
        
        # Scenario Analysis
        st.subheader("Scenario Analysis")
        
        scenarios = {
            "Conservative": {"capture": capture_rate * 0.5, "description": "Minimal marketing investment"},
            "Moderate": {"capture": capture_rate, "description": "Current trajectory"},
            "Aggressive": {"capture": capture_rate * 2, "description": "Significant marketing investment"},
        }
        
        scenario_cols = st.columns(3)
        for i, (scenario_name, scenario) in enumerate(scenarios.items()):
            with scenario_cols[i]:
                scenario_capture = scenario['capture'] / 100
                scenario_3yr = current_enrollment + int(total_k12_pool * scenario_capture * 0.75)  # 75% ramp by year 3
                st.metric(scenario_name, f"{scenario_3yr:,} (3yr)")
                st.caption(scenario['description'])
        
        # Action Items
        st.divider()
        st.subheader("Recommended Actions")
        
        if high_priority_pool > 0:
            st.markdown(f"""
            1. **Priority Marketing Campaign** - Target the **{len(df[df['zone'] == 'High Priority'])}** High Priority neighborhoods 
               with direct mail and digital ads. Estimated reach: **{high_priority_pool:,.0f}** families.
            
            2. **Open House Events** - Host campus tours at times convenient for commuters 
               (average commute: **{df['commute_minutes'].mean():.0f} min**).
            
            3. **Financial Aid Outreach** - For **{len(df[df['zone'] == 'Underserved'])}** Underserved areas, 
               emphasize scholarship and financial aid options.
            
            4. **Transportation Solutions** - Consider bus routes or carpool programs for areas 
               with commutes over 30 minutes.
            
            5. **Referral Program** - Leverage current families in high-performing areas to 
               recruit neighbors.
            """)
        else:
            st.info("Adjust filters to identify High Priority areas for targeted recommendations.")

if __name__ == "__main__":
    main()
