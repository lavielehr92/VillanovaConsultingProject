"""
Census Block Group Data Fetcher for Philadelphia
Downloads block group boundaries and demographic data
"""

import geopandas as gpd
import pandas as pd
import requests
import json
from urllib.parse import urlencode

def download_philadelphia_block_groups():
    """
    Download Census block group shapefiles and demographic data for Philadelphia County
    """
    print("Downloading Philadelphia County block group boundaries...")
    
    # Philadelphia County FIPS code: 42101 (PA state 42, Philadelphia County 101)
    state_fips = "42"
    county_fips = "101"
    
    try:
        # Download block group boundaries from Census TIGER/Line shapefiles
        # Using the web interface URL for 2022 block groups
        bg_url = f"https://www2.census.gov/geo/tiger/TIGER2022/BG/tl_2022_{state_fips}_bg.zip"
        
        print(f"Fetching from: {bg_url}")
        
        # Read the shapefile directly from URL
        gdf = gpd.read_file(bg_url)
        
        # Filter for Philadelphia County only
        phila_bg = gdf[gdf['COUNTYFP'] == county_fips].copy()
        
        print(f"Found {len(phila_bg)} block groups in Philadelphia County")
        
        # Create GEOID for matching with Census data
        phila_bg['GEOID'] = phila_bg['STATEFP'] + phila_bg['COUNTYFP'] + phila_bg['TRACTCE'] + phila_bg['BLKGRPCE']
        
        # Get demographic data from Census API for these block groups
        print("Fetching demographic data from Census API...")
        
        # Census variables we want
        variables = {
            'B01003_001E': 'total_pop',        # Total population
            'B08303_001E': 'total_commuters',  # Total commuters (for school age proxy)
            'B25077_001E': 'median_home_value', # Median home value
            'B19013_001E': 'median_income',     # Median household income
            'B25003_001E': 'total_housing',     # Total housing units
            'B25003_002E': 'owner_occupied',    # Owner occupied housing
            'B15003_001E': 'total_education',   # Total population 25+ for education
            'B15003_002E': 'no_schooling',      # No schooling completed
            'B15003_016E': 'high_school',       # High school graduate
            'B15003_021E': 'bachelor_degree',   # Bachelor's degree
            'B09001_001E': 'total_pop_age',     # Total population for age calculation
            'B09001_004E': 'age_5_9',           # Age 5-9
            'B09001_005E': 'age_10_14',         # Age 10-14
            'B09001_006E': 'age_15_17',         # Age 15-17
        }
        
        # Build Census API URL
        base_url = "https://api.census.gov/data/2022/acs/acs5"
        
        # Get all block group GEOIDs for the API call
        geoids = phila_bg['GEOID'].tolist()
        
        # Census API has limits, so we'll do this in batches
        all_census_data = []
        batch_size = 50  # Conservative batch size
        
        for i in range(0, len(geoids), batch_size):
            batch_geoids = geoids[i:i+batch_size]
            
            params = {
                'get': ','.join(variables.keys()),
                'for': f'block group:*',
                'in': f'state:{state_fips} county:{county_fips}',
                'key': 'YOUR_API_KEY_HERE'  # You'll need to get a free Census API key
            }
            
            # Remove key if not available (will use without key, but with rate limits)
            if params['key'] == 'YOUR_API_KEY_HERE':
                del params['key']
            
            try:
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    all_census_data.extend(data[1:])  # Skip header row
                    print(f"Fetched batch {i//batch_size + 1}/{(len(geoids)-1)//batch_size + 1}")
                else:
                    print(f"API request failed with status {response.status_code}")
                    break
            except Exception as e:
                print(f"Error fetching Census data: {e}")
                break
        
        if all_census_data:
            # Convert to DataFrame
            headers = list(variables.keys()) + ['state', 'county', 'tract', 'block_group']
            census_df = pd.DataFrame(all_census_data, columns=headers)
            
            # Create GEOID for merging
            census_df['GEOID'] = (census_df['state'] + census_df['county'] + 
                                census_df['tract'] + census_df['block_group'])
            
            # Rename columns to friendly names
            for var_code, friendly_name in variables.items():
                if var_code in census_df.columns:
                    census_df[friendly_name] = pd.to_numeric(census_df[var_code], errors='coerce')
            
            # Calculate derived variables
            census_df['k12_pop'] = (census_df.get('age_5_9', 0) + 
                                  census_df.get('age_10_14', 0) + 
                                  census_df.get('age_15_17', 0))
            
            census_df['poverty_rate'] = 0  # Placeholder - would need additional API call for poverty data
            census_df['pct_lt_hs'] = ((census_df.get('no_schooling', 0)) / 
                                    census_df.get('total_education', 1) * 100)
            
            # Merge with geographical data
            phila_bg_with_data = phila_bg.merge(census_df[['GEOID', 'total_pop', 'median_income', 
                                                         'k12_pop', 'poverty_rate', 'pct_lt_hs']], 
                                              on='GEOID', how='left')
        else:
            print("No Census data available, using geographical boundaries only")
            phila_bg_with_data = phila_bg.copy()
            
            # Create placeholder demographic data
            phila_bg_with_data['total_pop'] = 1000  # Placeholder
            phila_bg_with_data['median_income'] = 50000  # Placeholder
            phila_bg_with_data['k12_pop'] = 150  # Placeholder
            phila_bg_with_data['poverty_rate'] = 15.0  # Placeholder
            phila_bg_with_data['pct_lt_hs'] = 8.0  # Placeholder
        
        # Add estimated Christian and first-gen percentages (would need survey data for accuracy)
        phila_bg_with_data['pct_christian'] = 25.0  # Philadelphia average estimate
        phila_bg_with_data['pct_first_gen'] = 35.0  # First-generation estimate
        
        # Calculate centroid coordinates for compatibility with existing code
        phila_bg_with_data['centroid'] = phila_bg_with_data.geometry.centroid
        phila_bg_with_data['lat'] = phila_bg_with_data['centroid'].y
        phila_bg_with_data['lon'] = phila_bg_with_data['centroid'].x
        
        # Save the data
        print("Saving block group data...")
        
        # Save as GeoJSON for Plotly compatibility
        phila_bg_with_data.to_file("philadelphia_block_groups.geojson", driver="GeoJSON")
        
        # Save demographics as CSV for compatibility with existing code
        demo_columns = ['GEOID', 'median_income', 'k12_pop', 'poverty_rate', 'pct_lt_hs', 
                       'pct_christian', 'pct_first_gen', 'lat', 'lon', 'total_pop']
        
        demographics_bg = phila_bg_with_data[demo_columns].copy()
        demographics_bg.rename(columns={'GEOID': 'block_group_id', 
                                      'median_income': 'income',
                                      'pct_christian': '%Christian',
                                      'pct_first_gen': '%first_gen'}, inplace=True)
        
        demographics_bg.to_csv("demographics_block_groups.csv", index=False)
        
        print(f"✅ Successfully saved {len(phila_bg_with_data)} Philadelphia block groups")
        print("Files created:")
        print("  - philadelphia_block_groups.geojson (boundaries)")
        print("  - demographics_block_groups.csv (demographic data)")
        
        return phila_bg_with_data
        
    except Exception as e:
        print(f"❌ Error downloading block group data: {e}")
        print("Creating sample data for development...")
        
        # Create sample block group data for development
        sample_data = create_sample_block_group_data()
        return sample_data

def create_sample_block_group_data():
    """
    Create sample block group data for development/testing
    """
    print("Creating sample block group data...")
    
    # Create some sample block groups around Philadelphia
    sample_blocks = []
    
    # Define some areas around Philadelphia with sample boundaries
    base_lat, base_lon = 39.952, -75.193
    
    for i in range(20):  # Create 20 sample block groups
        # Create simple rectangular boundaries
        lat_offset = (i % 4) * 0.01 - 0.015
        lon_offset = (i // 4) * 0.01 - 0.025
        
        center_lat = base_lat + lat_offset
        center_lon = base_lon + lon_offset
        
        # Create a simple rectangular polygon
        from shapely.geometry import Polygon
        
        polygon = Polygon([
            (center_lon - 0.004, center_lat - 0.004),
            (center_lon + 0.004, center_lat - 0.004),
            (center_lon + 0.004, center_lat + 0.004),
            (center_lon - 0.004, center_lat + 0.004)
        ])
        
        sample_blocks.append({
            'GEOID': f'4210100{i:04d}1',  # Sample GEOID format
            'geometry': polygon,
            'total_pop': np.random.randint(800, 2000),
            'median_income': np.random.randint(35000, 85000),
            'k12_pop': np.random.randint(100, 400),
            'poverty_rate': np.random.uniform(5, 35),
            'pct_lt_hs': np.random.uniform(3, 15),
            'pct_christian': np.random.uniform(15, 45),
            'pct_first_gen': np.random.uniform(20, 60),
            'lat': center_lat,
            'lon': center_lon
        })
    
    import numpy as np
    gdf = gpd.GeoDataFrame(sample_blocks, crs="EPSG:4326")
    
    # Save sample data
    gdf.to_file("philadelphia_block_groups.geojson", driver="GeoJSON")
    
    demo_columns = ['GEOID', 'median_income', 'k12_pop', 'poverty_rate', 'pct_lt_hs', 
                   'pct_christian', 'pct_first_gen', 'lat', 'lon', 'total_pop']
    
    demographics_bg = gdf[demo_columns].copy()
    demographics_bg.rename(columns={'GEOID': 'block_group_id', 
                                  'median_income': 'income',
                                  'pct_christian': '%Christian',
                                  'pct_first_gen': '%first_gen'}, inplace=True)
    
    demographics_bg.to_csv("demographics_block_groups.csv", index=False)
    
    print("✅ Sample data created successfully")
    return gdf

if __name__ == "__main__":
    import numpy as np
    download_philadelphia_block_groups()