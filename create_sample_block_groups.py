"""
Create sample block group data for testing the choropleth visualization
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import json

def create_sample_philadelphia_block_groups():
    """
    Create sample Philadelphia block group boundaries and data for testing
    """
    
    # Philadelphia bounds approximately
    phila_bounds = {
        'north': 40.138,
        'south': 39.867,  
        'east': -74.955,
        'west': -75.280
    }
    
    # Create a grid of block groups
    lat_step = (phila_bounds['north'] - phila_bounds['south']) / 15  # 15 rows
    lon_step = (phila_bounds['east'] - phila_bounds['west']) / 20   # 20 columns
    
    block_groups = []
    demographics_data = []
    
    block_id = 1
    
    for i in range(15):  # 15 rows
        for j in range(20):  # 20 columns
            
            # Calculate bounds for this block group
            south = phila_bounds['south'] + i * lat_step
            north = south + lat_step
            west = phila_bounds['west'] + j * lon_step  
            east = west + lon_step
            
            # Create rectangular polygon
            polygon = Polygon([
                (west, south),
                (east, south), 
                (east, north),
                (west, north),
                (west, south)
            ])
            
            # Create GEOID (Census format: state+county+tract+blockgroup)
            geoid = f'4210100{i:02d}{j:02d}1'
            
            # Calculate center point
            center_lat = (north + south) / 2
            center_lon = (east + west) / 2
            
            # Generate realistic demographic data
            # Base values with some geographic patterns
            distance_from_center = np.sqrt((center_lat - 39.952)**2 + (center_lon + 75.193)**2)
            
            # Income tends to be higher in certain areas, lower in others
            base_income = 45000
            if center_lat > 40.0:  # Northern areas tend to have higher income
                income_mult = 1.4
            elif center_lon > -75.15:  # Eastern areas
                income_mult = 1.2
            else:
                income_mult = 0.8 + np.random.uniform(0, 0.6)
            
            income = int(base_income * income_mult * (0.8 + np.random.uniform(0, 0.4)))
            income = max(25000, min(income, 150000))  # Reasonable bounds
            
            # K-12 population varies by density
            base_k12 = 150
            density_factor = 1.5 - distance_from_center * 20  # Closer to center = more dense
            k12_pop = int(base_k12 * max(0.3, density_factor) * (0.7 + np.random.uniform(0, 0.6)))
            k12_pop = max(50, min(k12_pop, 500))
            
            # Poverty rate inversely related to income
            poverty_rate = max(2.0, min(45.0, 50.0 - (income - 25000) / 2500))
            
            # Education levels related to income
            pct_lt_hs = max(1.0, min(25.0, 30.0 - (income - 25000) / 5000))
            
            # Christian percentage - varies by area
            pct_christian = np.random.uniform(15, 50)
            
            # First generation percentage - higher in lower income areas
            pct_first_gen = max(10, min(70, 60 - (income - 25000) / 2000))
            
            block_groups.append({
                'STATEFP': '42',
                'COUNTYFP': '101', 
                'TRACTCE': f'{i:02d}{j:02d}',
                'BLKGRPCE': '1',
                'GEOID': geoid,
                'geometry': polygon
            })
            
            demographics_data.append({
                'block_group_id': geoid,
                'income': income,
                'k12_pop': k12_pop,
                'poverty_rate': round(poverty_rate, 1),
                'pct_lt_hs': round(pct_lt_hs, 1),
                '%Christian': round(pct_christian, 1),
                '%first_gen': round(pct_first_gen, 1),
                'lat': center_lat,
                'lon': center_lon,
                'total_pop': k12_pop * 4  # Rough estimate of total population
            })
            
            block_id += 1
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(block_groups, crs='EPSG:4326')
    
    # Create demographics DataFrame  
    demographics_df = pd.DataFrame(demographics_data)
    
    # Save files
    print(f"Creating {len(gdf)} sample block groups...")
    
    gdf.to_file("philadelphia_block_groups.geojson", driver="GeoJSON")
    demographics_df.to_csv("demographics_block_groups.csv", index=False)
    
    print("✅ Sample block group data created successfully!")
    print("Files created:")
    print("  - philadelphia_block_groups.geojson")
    print("  - demographics_block_groups.csv")
    
    # Print some stats
    print(f"\nData summary:")
    print(f"  - Block groups: {len(gdf)}")
    print(f"  - Income range: ${demographics_df['income'].min():,} - ${demographics_df['income'].max():,}")
    print(f"  - K-12 population range: {demographics_df['k12_pop'].min()} - {demographics_df['k12_pop'].max()}")
    print(f"  - Total K-12 students: {demographics_df['k12_pop'].sum():,}")
    
    return gdf, demographics_df

if __name__ == "__main__":
    import geopandas as gpd
    create_sample_philadelphia_block_groups()