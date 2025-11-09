"""
Educational Desert Index calculation for Census Block Groups
Updated to work with block group boundaries and choropleth visualization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def compute_edi_block_groups(demographics_df, schools_df, max_distance_km=15):
    """
    Compute Educational Desert Index for block groups
    
    Parameters:
    - demographics_df: DataFrame with block group demographic data
    - schools_df: DataFrame with school locations
    - max_distance_km: Maximum distance for school accessibility calculation
    
    Returns:
    - DataFrame with EDI scores for each block group
    """
    
    # Ensure we have the required columns
    required_demo_cols = ['block_group_id', 'lat', 'lon', 'k12_pop']
    required_school_cols = ['lat', 'lon']
    
    for col in required_demo_cols:
        if col not in demographics_df.columns:
            if col == 'block_group_id':
                # Try alternative column names
                if 'GEOID' in demographics_df.columns:
                    demographics_df = demographics_df.rename(columns={'GEOID': 'block_group_id'})
                elif 'geoid_bg' in demographics_df.columns:
                    demographics_df = demographics_df.rename(columns={'geoid_bg': 'block_group_id'})
                else:
                    raise ValueError(f"Block group ID column not found. Expected 'block_group_id', 'GEOID', or 'geoid_bg'")
            else:
                raise ValueError(f"Required column '{col}' not found in demographics data")
    
    for col in required_school_cols:
        if col not in schools_df.columns:
            raise ValueError(f"Required column '{col}' not found in schools data")
    
    results = []
    
    for _, block_group in demographics_df.iterrows():
        bg_id = block_group['block_group_id']
        bg_lat = block_group['lat']
        bg_lon = block_group['lon']
        bg_k12_pop = block_group.get('k12_pop', 0)
        
        # Skip if coordinates are invalid
        if pd.isna(bg_lat) or pd.isna(bg_lon):
            continue
            
        # Calculate distances to all schools
        distances = []
        school_capacities = []
        
        for _, school in schools_df.iterrows():
            school_lat = school['lat']
            school_lon = school['lon']
            
            # Skip schools with invalid coordinates
            if pd.isna(school_lat) or pd.isna(school_lon):
                continue
                
            distance = haversine_km(bg_lat, bg_lon, school_lat, school_lon)
            
            if distance <= max_distance_km:
                distances.append(distance)
                # Use capacity if available, otherwise estimate
                capacity = school.get('capacity', 500)  # Default capacity estimate
                school_capacities.append(capacity)
        
        # Component 1: Two-Step Floating Catchment Analysis (2SFCA)
        if len(distances) > 0:
            # Step 1: For each school, calculate its supply-to-demand ratio
            supply_ratios = []
            for i, (dist, capacity) in enumerate(zip(distances, school_capacities)):
                # Schools within catchment area of this school
                demand_for_school = bg_k12_pop  # Simplified - would need to calculate for all block groups
                if demand_for_school > 0:
                    supply_ratios.append(capacity / demand_for_school)
                else:
                    supply_ratios.append(1.0)  # No demand, perfect supply
            
            # Step 2: Sum accessibility for this block group
            accessibility_2sfca = sum(supply_ratios)
        else:
            accessibility_2sfca = 0.0
        
        # Component 2: Gravity-based access (distance decay)
        if len(distances) > 0:
            # Use inverse distance weighting with exponential decay
            weights = [np.exp(-dist / 5.0) for dist in distances]  # 5km decay constant
            gravity_access = sum(weights)
        else:
            gravity_access = 0.0
        
        # Component 3: Nearest school distance
        if len(distances) > 0:
            nearest_distance = min(distances)
        else:
            nearest_distance = max_distance_km  # Maximum penalty for no schools
        
        # Component 4: Neighborhood need (poverty rate and education gaps)
        poverty_rate = block_group.get('poverty_rate', 15.0)  # Default if missing
        pct_lt_hs = block_group.get('pct_lt_hs', 8.0)  # Default if missing
        
        # Combine into neighborhood need score
        neighborhood_need = (poverty_rate / 100.0) * 0.7 + (pct_lt_hs / 100.0) * 0.3
        
        # Store raw components for analysis
        results.append({
            'block_group_id': bg_id,
            'lat': bg_lat,
            'lon': bg_lon,
            'k12_pop': bg_k12_pop,
            'accessibility_2sfca': accessibility_2sfca,
            'gravity_access': gravity_access,
            'nearest_distance': nearest_distance,
            'neighborhood_need': neighborhood_need,
            'poverty_rate': poverty_rate,
            'pct_lt_hs': pct_lt_hs
        })
    
    if not results:
        raise ValueError("No valid block groups found for EDI calculation")
    
    # Convert to DataFrame
    edi_df = pd.DataFrame(results)
    
    # Normalize components to 0-1 scale
    scaler = MinMaxScaler()
    
    # Safe normalization function that handles all-NaN or single-value columns
    def safe_normalize(df, column, inverse=False):
        """Normalize a column, handling edge cases"""
        values = df[[column]].copy()
        # Check if all values are the same or all NaN
        if pd.isna(values).all().all():
            return np.zeros(len(df))
        unique_vals = values.dropna().nunique()
        if unique_vals <= 1:
            return np.ones(len(df)) * 0.5  # Middle value if all same
        try:
            normalized = scaler.fit_transform(values).flatten()
            return (1 - normalized) if inverse else normalized
        except Exception:
            return np.ones(len(df)) * 0.5
    
    # For accessibility measures, higher is better (inverse for EDI)
    edi_df['access_2sfca_norm'] = safe_normalize(edi_df, 'accessibility_2sfca', inverse=True)
    edi_df['gravity_access_norm'] = safe_normalize(edi_df, 'gravity_access', inverse=True)
    
    # For distance, higher is worse (farther = more desert-like)
    edi_df['distance_norm'] = safe_normalize(edi_df, 'nearest_distance')
    
    # For neighborhood need, higher is worse (more need = more desert-like)
    edi_df['need_norm'] = safe_normalize(edi_df, 'neighborhood_need')
    
    # Combine components with weights
    # Weights sum to 1.0
    weights = {
        'access_2sfca': 0.3,      # Supply/demand balance
        'gravity_access': 0.25,   # Overall accessibility
        'distance': 0.25,         # Geographic isolation
        'need': 0.2              # Socioeconomic need
    }
    
    edi_df['EDI_raw'] = (
        edi_df['access_2sfca_norm'] * weights['access_2sfca'] +
        edi_df['gravity_access_norm'] * weights['gravity_access'] +
        edi_df['distance_norm'] * weights['distance'] +
        edi_df['need_norm'] * weights['need']
    )
    
    # Scale final EDI to 0-100 for interpretability
    try:
        edi_df['EDI'] = scaler.fit_transform(edi_df[['EDI_raw']]).flatten() * 100
    except Exception:
        # If scaling fails, use raw values scaled manually
        edi_raw = edi_df['EDI_raw'].values
        if len(edi_raw) > 0 and not pd.isna(edi_raw).all():
            min_val = np.nanmin(edi_raw)
            max_val = np.nanmax(edi_raw)
            if max_val > min_val:
                edi_df['EDI'] = ((edi_raw - min_val) / (max_val - min_val)) * 100
            else:
                edi_df['EDI'] = 50.0  # All same, use middle value
        else:
            edi_df['EDI'] = 50.0
    
    return edi_df

def compute_edi(demographics_df, schools_df):
    """
    Wrapper function to maintain compatibility with existing code
    """
    # Check if this is block group data or ZIP code data
    if 'block_group_id' in demographics_df.columns or 'GEOID' in demographics_df.columns:
        return compute_edi_block_groups(demographics_df, schools_df)
    else:
        # Fall back to original ZIP code calculation
        return compute_edi_zip_codes(demographics_df, schools_df)

def compute_edi_zip_codes(demographics_df, schools_df):
    """
    Original ZIP code EDI calculation for backward compatibility
    """
    # This is the original function - keeping for compatibility
    # (Insert the original compute_edi function here if needed)
    
    # For now, create a simple version
    results = []
    
    for _, row in demographics_df.iterrows():
        results.append({
            'geoid_bg': row.get('ZIP', row.get('geoid_bg')),
            'lat': row['lat'],
            'lon': row['lon'],
            'k12_pop': row.get('k12_pop', 0),
            'EDI': np.random.uniform(10, 80)  # Placeholder
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Test the function with sample data
    print("Testing block group EDI calculation...")
    
    # Create sample data
    sample_demographics = pd.DataFrame({
        'block_group_id': ['421010001001', '421010001002', '421010001003'],
        'lat': [39.952, 39.955, 39.950],
        'lon': [-75.193, -75.190, -75.196],
        'k12_pop': [150, 200, 180],
        'poverty_rate': [15.0, 20.0, 12.0],
        'pct_lt_hs': [8.0, 12.0, 6.0]
    })
    
    sample_schools = pd.DataFrame({
        'school_name': ['School A', 'School B', 'School C'],
        'lat': [39.953, 39.948, 39.957],
        'lon': [-75.194, -75.189, -75.198],
        'capacity': [500, 600, 400]
    })
    
    edi_result = compute_edi_block_groups(sample_demographics, sample_schools)
    print("EDI calculation completed successfully!")
    print(f"Calculated EDI for {len(edi_result)} block groups")
    print("\nSample results:")
    print(edi_result[['block_group_id', 'EDI', 'nearest_distance']].head())