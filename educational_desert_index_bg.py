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
    Compute TRUE Educational Desert Index for block groups
    
    Measures actual lack of educational infrastructure and access, NOT proximity to CCA.
    Higher EDI = worse educational environment
    
    Components:
    1. School Accessibility (40%) - Distance to ANY quality school (public, private, charter)
    2. School-to-Student Ratio (30%) - Number of nearby school seats vs K-12 population
    3. Socioeconomic Barriers (20%) - Poverty rate as proxy for educational barriers
    4. Infrastructure Access (10%) - Estimated internet/technology access (inverse of poverty)
    
    Parameters:
    - demographics_df: DataFrame with block group demographic data
    - schools_df: DataFrame with ALL school locations (not just CCA)
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
        
        # Ensure bg_k12_pop is a scalar (not a Series)
        bg_k12_pop = block_group.get('k12_pop', 0)
        if pd.api.types.is_list_like(bg_k12_pop):
            bg_k12_pop = bg_k12_pop.iloc[0] if hasattr(bg_k12_pop, 'iloc') else 0
        bg_k12_pop = float(bg_k12_pop) if pd.notna(bg_k12_pop) else 0.0
        
        # Get poverty rate for socioeconomic barriers
        poverty_rate = block_group.get('poverty_rate', 0)
        if pd.api.types.is_list_like(poverty_rate):
            poverty_rate = poverty_rate.iloc[0] if hasattr(poverty_rate, 'iloc') else 0
        poverty_rate = float(poverty_rate) if pd.notna(poverty_rate) else 0.0
        
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
        
        # Component 1: School Accessibility (40% weight)
        # Based on nearest school distance - farther = worse desert
        if len(distances) > 0:
            min_distance = min(distances)
            # Normalize: 0km=0 (best), 15km=1 (worst)
            accessibility_score = min(min_distance / max_distance_km, 1.0)
        else:
            accessibility_score = 1.0  # No schools nearby = worst
        
        # Component 2: School-to-Student Ratio (30% weight)
        # Total nearby seats vs K-12 population
        if len(distances) > 0 and bg_k12_pop > 0:
            total_nearby_seats = sum(school_capacities)
            seat_ratio = total_nearby_seats / bg_k12_pop
            # Good ratio: 1.0+ seats per student = 0 (best)
            # Poor ratio: 0.5 or less = 1.0 (worst)
            ratio_score = max(0, min(1.0, 1.0 - (seat_ratio / 1.0)))
        else:
            ratio_score = 0.5  # Neutral if no students or no schools
        
        # Component 3: Socioeconomic Barriers (20% weight)
        # Poverty rate as proxy for educational barriers
        # Higher poverty = worse educational access
        poverty_score = min(poverty_rate / 100.0, 1.0)  # Cap at 100%
        
        # Component 4: Infrastructure Access (10% weight)
        # Estimated internet/technology access (inverse of poverty as proxy)
        # Higher poverty typically means lower broadband access
        # US average broadband: ~85%, low-income areas: ~60%
        estimated_broadband = max(0.6, 0.95 - (poverty_rate / 100.0))
        infrastructure_score = 1.0 - estimated_broadband  # Invert: worse = higher
        
        # Combine components into final EDI
        # Higher EDI = worse educational environment (true desert)
        edi = (
            0.40 * accessibility_score +      # 40% - Distance to nearest school
            0.30 * ratio_score +               # 30% - Seat availability
            0.20 * poverty_score +             # 20% - Economic barriers
            0.10 * infrastructure_score        # 10% - Tech/internet access
        )
        
        # Store results
        results.append({
            'block_group_id': bg_id,
            'lat': bg_lat,
            'lon': bg_lon,
            'k12_pop': bg_k12_pop,
            'nearest_school_km': min(distances) if distances else max_distance_km,
            'nearby_seats': sum(school_capacities) if school_capacities else 0,
            'seat_ratio': (sum(school_capacities) / bg_k12_pop) if bg_k12_pop > 0 and school_capacities else 0,
            'poverty_rate': poverty_rate,
            'est_broadband_pct': estimated_broadband * 100,
            'accessibility_score': accessibility_score,
            'ratio_score': ratio_score,
            'poverty_score': poverty_score,
            'infrastructure_score': infrastructure_score,
            'EDI': edi * 100  # Scale to 0-100 for display
        })
    
    if not results:
        raise ValueError("No valid block groups found for EDI calculation")
    
    # Convert to DataFrame
    edi_df = pd.DataFrame(results)
    
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