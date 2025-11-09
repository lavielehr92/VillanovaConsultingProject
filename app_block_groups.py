"""
Philadelphia Educational Desert Explorer - Block Group Version
CCA Expansion Analysis Dashboard with Choropleth Visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import json
import os
from educational_desert_index_bg import compute_edi_block_groups, haversine_km
import numpy as np
try:
    # Allow optional live refresh import
    from fetch_block_groups_live import main as fetch_live_block_groups
except Exception:
    fetch_live_block_groups = None

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

# Page config
st.set_page_config(
    page_title="Philadelphia Educational Desert Explorer - Block Groups", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #d4d4d4;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour, then reload
def load_block_group_data():
    """Load and cache block group data"""
    try:
        # Try to load the block group data
        gdf = gpd.read_file('philadelphia_block_groups.geojson')
        demographics = pd.read_csv('demographics_block_groups.csv')
        
        # FIX: Ensure GEOID formats match for proper merging
        # CSV has float IDs like "421019809016.0", GeoJSON has int/string like "421019809016"
        # Convert both to integer then to string to ensure they match
        gdf['GEOID'] = gdf['GEOID'].astype(str).str.replace('.0', '', regex=False)
        demographics['block_group_id'] = demographics['block_group_id'].astype(float).astype(int).astype(str)
        
        # CRITICAL: Clean Census sentinel values BEFORE any calculations
        # Census uses -666666666 to indicate "no data available"
        sentinel_cols = ['income', 'poverty_rate', 'total_pop', 'pct_black', 'pct_white', 'hh_with_u18']
        for col in sentinel_cols:
            if col in demographics.columns:
                demographics[col] = demographics[col].replace(-666666666, pd.NA)
                demographics[col] = pd.to_numeric(demographics[col], errors='coerce')
        
        return gdf, demographics
    except Exception as e:
        st.error(f"Error loading block group data: {e}")
        st.info("Please run fetch_block_groups.py first to download Census block group data")
        st.stop()

@st.cache_data  
def load_other_data():
    """Load schools and student data"""
    try:
        schools = pd.read_csv('schools.csv')
        current_students = pd.read_csv('current_students.csv')
        return schools, current_students
    except Exception as e:
        st.error(f"Error loading supporting data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def create_choropleth_map(gdf_filtered, demographics_filtered, color_column, title, show_students=False, students_df=None):
    """Create choropleth map with block group boundaries"""
    
    # Debug output
    st.sidebar.write(f"🔍 Debug: Creating map with {len(gdf_filtered)} geographic features")
    st.sidebar.write(f"🔍 Debug: Demographic data has {len(demographics_filtered)} records")
    
    # Merge geodata with demographic data
    plot_data = gdf_filtered.merge(demographics_filtered, left_on='GEOID', right_on='block_group_id', how='left')
    
    st.sidebar.write(f"🔍 Debug: Merged data has {len(plot_data)} rows")
    
    if len(plot_data) == 0:
        st.error("❌ No data to display on map after filtering!")
        return go.Figure()
    
    # Check if color_column exists
    if color_column not in plot_data.columns:
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
            feature['properties']['k12_pop'] = int(row.get('k12_pop', 0))
            feature['properties']['income'] = int(row.get('income', 0))
            feature['properties']['poverty_rate'] = float(row.get('poverty_rate', 0))
            if color_column in row:
                feature['properties'][color_column] = float(row.get(color_column, 0))
    
    # Build custom hover template with proper formatting for missing data
    # Note: NaN values will display as "nan" in plotly, we'll handle this in customdata
    hover_template = '<b>Block Group: %{customdata[0]}</b><br>'
    hover_template += 'Tract: %{customdata[1]}<br>'
    hover_template += '<b>K-12 Population: %{customdata[2]}</b><br>'
    hover_template += 'Median Income: %{customdata[3]}<br>'
    hover_template += 'Poverty Rate: %{customdata[4]}<br>'
    hover_template += 'Total Pop: %{customdata[5]}<br>'
    hover_template += 'HH with Children <18: %{customdata[6]}<br>'
    hover_template += '% First-Gen College: %{customdata[7]}<br>'
    hover_template += '% Christian: %{customdata[8]}<br>'
    hover_template += f'<b>{color_column.replace("_", " ").title()}: %{{customdata[9]}}</b><extra></extra>'
    
    # Prepare custom data for hover with proper formatting
    cleaned = plot_data.copy()
    
    # Clean sentinel values
    sentinel_cols = ['income', 'poverty_rate', 'pct_black', 'pct_white', '%first_gen', '%Christian']
    for col in sentinel_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].replace(-666666666, np.nan)
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
    
    # Ensure color column sentinel removed
    if color_column in cleaned.columns:
        cleaned[color_column] = cleaned[color_column].replace(-666666666, np.nan)
    
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
        else:
            return f'{val:.1f}'
    
    # Create formatted customdata
    customdata_list = []
    for idx, row in cleaned.iterrows():
        customdata_list.append([
            str(row.get('block_group_id', 'N/A')),
            str(row.get('TRACTCE', 'N/A')),
            format_value(row, 'k12_pop', 'integer'),
            format_value(row, 'income', 'currency'),
            format_value(row, 'poverty_rate', 'percent'),
            format_value(row, 'total_pop', 'integer'),
            format_value(row, 'hh_with_u18', 'integer'),
            format_value(row, '%first_gen', 'percent'),
            format_value(row, '%Christian', 'percent'),
            format_value(row, color_column, 'number' if color_column == 'EDI' else 'integer')
        ])
    
    customdata = np.array(customdata_list)
    
    # Get z values for coloring
    z_vals = cleaned[color_column].values if color_column in cleaned.columns else plot_data[color_column].values
    # Fallback if all-NaN z values
    if np.all(np.isnan(z_vals)):
        fallback_col = 'k12_pop' if 'k12_pop' in cleaned.columns else None
        if fallback_col:
            st.warning(f"No valid values in '{color_column}' to color by; falling back to '{fallback_col}'.")
            color_column = fallback_col
            z_vals = safe_arr(color_column)
        else:
            st.error("No valid data available for coloring the map.")
            return go.Figure()
    
    # Create the choropleth using graph_objects for better control
    fig = go.Figure()
    
    # Add choropleth trace
    fig.add_choroplethmapbox(
        geojson=geojson_data,
        locations=plot_data.index,
        z=z_vals,
        featureidkey="id",  # Link to the 'id' we set in the GeoJSON features
        customdata=customdata,
        colorscale="RdYlBu_r",
        marker_opacity=0.7,
        marker_line_width=1,
        marker_line_color='white',
        hovertemplate=hover_template,
        colorbar=dict(
            title=color_column.replace('_', ' ').title(),
            len=0.7,
            x=1.02
        ),
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
        
        # First generation percentage (higher is better for mission)
        first_gen = bg.get('%first_gen', 0)
        if first_gen >= 40:
            score += 2
        elif first_gen >= 25:
            score += 1
        
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
    st.title("🏫 Philadelphia Educational Desert Explorer")
    st.subheader("Block Group Analysis for CCA Expansion Planning")
    
    # Add explanation of EDI in an expandable section
    with st.expander("ℹ️ What is the Educational Desert Index (EDI)?"):
        st.markdown("""
        ### Educational Desert Index (EDI) - Scale: 0 to 100
        
        The **EDI score** measures how underserved a neighborhood is for quality educational options. 
        
        **Higher EDI = Greater Educational Need** (more "desert-like")
        
        #### EDI Components:
        1. **School Accessibility (55%)**: Distance and capacity of nearby schools
           - Fewer nearby schools → Higher EDI
           - Schools at capacity → Higher EDI
        
        2. **Geographic Isolation (25%)**: Physical distance to nearest quality school
           - Farther from schools → Higher EDI
        
        3. **Neighborhood Need (20%)**: Socioeconomic indicators
           - Higher poverty rates → Higher EDI
           - Lower educational attainment → Higher EDI
        
        #### How to Use EDI:
        - **EDI 70-100**: High priority areas with severe educational gaps
        - **EDI 40-69**: Moderate need areas worth consideration
        - **EDI 0-39**: Well-served areas with adequate school access
        
        Block groups with high EDI scores and high K-12 populations are prime expansion targets.
        
        #### About the Data:
        - **K-12 Population**: Census ages 5-17 from ACS 2022
        - **0-Population Block Groups**: ~5% of block groups have 0 residents (parks, water, industrial areas, etc.) - these are excluded from EDI by default
        - **Median Income**: Per block group (not household filtering)
        - **Poverty Rate**: Percentage of population below federal poverty line
        - **Special Tract Codes**: Tracts starting with 98xxxx are typically non-residential (water bodies, parks, large facilities)
        """)
    
    # Load data
    with st.spinner("Loading Census block group data..."):
        gdf, demographics = load_block_group_data()
        schools, current_students = load_other_data()
    
    st.success(f"✅ Loaded {len(gdf)} block groups and {len(schools)} schools")
    
    # Add helpful info about the data
    with st.expander("ℹ️ About This Data", expanded=False):
        st.write(f"""
        **Philadelphia Block Group Data:**
        - Total block groups: {len(demographics):,}
        - Total K-12 population: {demographics['k12_pop'].sum():,.0f}
        - Block groups with students: {(demographics['k12_pop'] > 0).sum():,}
        - Non-residential areas (0 pop): {(demographics['total_pop'] == 0).sum():,}
        
        **Note:** Many block groups have 0 K-12 students because they are parks, commercial zones, 
        industrial areas, or retirement communities. Use the "Optional Filters" to hide these areas.
        """)
    
    # Sidebar filters
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Proximity filters
    st.sidebar.subheader("Geographic Filters")
    max_distance = st.sidebar.slider(
        "Max Distance from CCA Campuses (km)", 
        1, 35, 15, 
        help="Show only block groups within this distance of CCA campuses"
    )
    
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
    
    # Demographic filters
    st.sidebar.subheader("Demographic Indicators")
    st.sidebar.caption("Median income shown is per block group (not a hard household cutoff). All household income levels are represented inside each block group's median value.")

    # Optional live data refresh (Census API)
    census_api_key = get_census_api_key()
    if fetch_live_block_groups and census_api_key:
        with st.sidebar.expander("🔄 Live Census Data"):
            st.write("You have a Census API key loaded. You can pull fresh ACS data.")
            if st.button("Refresh From Census API", key="refresh_census"):
                with st.spinner("Pulling live ACS data (may take ~30s)..."):
                    try:
                        fetch_live_block_groups()
                        st.success("Live data downloaded. Please rerun (Ctrl+R) or click 'Rerun' above.")
                    except Exception as e:
                        st.error(f"Live fetch failed: {e}")
    
    if not demographics_filtered.empty:
        # Show income distribution but do NOT filter unless user opts in explicitly
        st.sidebar.markdown("**Income Distribution (Median per Block Group)**")
        valid_incomes = demographics_filtered['income'].dropna()
        if len(valid_incomes) > 0:
            min_income = int(valid_incomes.min())
            max_income = int(valid_incomes.max())
            st.sidebar.progress(0)
            st.sidebar.write(f"Range: ${min_income:,} – ${max_income:,}")
            enable_income_filter = st.sidebar.checkbox("Enable Median Income Filter (optional)", value=False)
            if enable_income_filter:
                income_range = st.sidebar.slider(
                    "Select Median Income Range", 
                    min_income,
                    max_income,
                    (min_income, max_income),
                    step=5000,
                    format="$%d",
                    key="income_range"
                )
                demographics_filtered = demographics_filtered[
                    (demographics_filtered['income'] >= income_range[0]) &
                    (demographics_filtered['income'] <= income_range[1])
                ]
                st.sidebar.info(f"Filtered to {len(demographics_filtered)} block groups in selected median range.")
            else:
                st.sidebar.success(f"All {len(demographics_filtered)} block groups included (no median income filtering).")
        else:
            st.sidebar.warning("⚠️ No valid income data available in filtered block groups")
        
        # Additional demographic filters (optional)
        st.sidebar.subheader("Optional Filters")
        
        # Filter out non-residential block groups
        hide_zero_pop = st.sidebar.checkbox(
            "Hide Non-Residential Areas (0 population)", 
            value=True,
            help="Excludes parks, water bodies, industrial zones, etc. with 0 population"
        )
        if hide_zero_pop:
            before_count = len(demographics_filtered)
            demographics_filtered = demographics_filtered[demographics_filtered['total_pop'] > 0].copy()
            removed = before_count - len(demographics_filtered)
            if removed > 0:
                st.sidebar.info(f"✓ Filtered out {removed} non-residential block groups")
            
            # Also show how many remain with 0 K-12 pop (residential but no children)
            zero_k12 = len(demographics_filtered[demographics_filtered['k12_pop'] == 0])
            if zero_k12 > 0:
                st.sidebar.caption(f"ℹ️ {zero_k12} residential blocks have 0 K-12 children")
        
        # Additional filter for 0 K-12 population
        hide_zero_k12 = st.sidebar.checkbox(
            "Also hide blocks with 0 K-12 children",
            value=False,
            help="Excludes residential areas with no school-age children (e.g., retirement communities, young professional areas)"
        )
        if hide_zero_k12:
            before_count = len(demographics_filtered)
            demographics_filtered = demographics_filtered[demographics_filtered['k12_pop'] > 0].copy()
            removed = before_count - len(demographics_filtered)
            if removed > 0:
                st.sidebar.info(f"✓ Filtered out {removed} additional blocks with 0 K-12 children")
        
        # Poverty rate filter
        apply_poverty_filter = st.sidebar.checkbox("Filter by Poverty Rate", value=False)
        if apply_poverty_filter:
            # Get valid poverty rates (exclude NaN values)
            valid_poverty = demographics_filtered['poverty_rate'].dropna()
            if len(valid_poverty) > 0:
                poverty_range = st.sidebar.slider(
                    "Poverty Rate Range (%)", 
                    float(valid_poverty.min()), 
                    float(valid_poverty.max()), 
                    (float(valid_poverty.min()), float(valid_poverty.max())),
                    step=1.0
                )
                demographics_filtered = demographics_filtered[
                    (demographics_filtered['poverty_rate'] >= poverty_range[0]) &
                    (demographics_filtered['poverty_rate'] <= poverty_range[1])
                ]
            else:
                st.sidebar.warning("No valid poverty rate data available")
        
        # Filter geodataframe to match
        gdf_filtered = gdf[gdf['GEOID'].isin(demographics_filtered['block_group_id'])]
        
    else:
        st.warning("No block groups found within the specified distance.")
        gdf_filtered = gdf.iloc[0:0]  # Empty dataframe
        
    # Map display options
    st.sidebar.subheader("Map Display")
    show_current_students = st.sidebar.checkbox("Show Current Students", value=False)
    
    # Visualization selection
    color_options = {
        'Educational Desert Index': 'EDI',
        'K-12 Population': 'k12_pop', 
        'Median Income': 'income',
        'Poverty Rate': 'poverty_rate',
        'Marketing Priority': 'marketing_priority'
    }
    
    selected_metric = st.sidebar.selectbox(
        "Color Map By:", 
        list(color_options.keys()),
        index=0
    )
    
    # Calculate EDI if needed
    if not demographics_filtered.empty and not schools.empty:
        with st.spinner("Calculating Educational Desert Index..."):
            try:
                # Only calculate EDI for block groups with actual population
                demographics_with_pop = demographics_filtered[demographics_filtered['total_pop'] > 0].copy()
                
                if len(demographics_with_pop) > 0:
                    edi_df = compute_edi_block_groups(demographics_with_pop, schools)
                    # Merge EDI back to all demographics (including 0-pop areas)
                    demographics_filtered = demographics_filtered.merge(
                        edi_df[['block_group_id', 'EDI']], 
                        on='block_group_id', 
                        how='left'
                    )
                    # Fill NaN EDI values (for 0-pop areas) with a low score
                    demographics_filtered = demographics_filtered.copy()
                    demographics_filtered['EDI'] = demographics_filtered['EDI'].fillna(0.0)
                else:
                    demographics_filtered = demographics_filtered.copy()
                    demographics_filtered['EDI'] = 0.0
            except Exception as e:
                st.sidebar.warning(f"⚠️ EDI calculation issue: {str(e)[:100]}")
                # Use a simple distance-based proxy for EDI
                demographics_filtered = demographics_filtered.copy()
                def simple_edi(row):
                    if row.get('total_pop', 0) == 0:
                        return 0.0  # No population = not a desert, just uninhabited
                    min_dist = min([haversine_km(row['lat'], row['lon'], c['lat'], c['lon']) 
                                   for _, c in cca_campuses.iterrows()])
                    return min(100, min_dist * 5)  # Simple distance-based score
                demographics_filtered['EDI'] = demographics_filtered.apply(simple_edi, axis=1)
        
        # Calculate marketing priority
        demographics_filtered = demographics_filtered.copy()
        demographics_filtered['marketing_priority'] = calculate_marketing_priority_bg(demographics_filtered, cca_campuses)
    
    # Main content area
    if not demographics_filtered.empty:
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Block Groups", 
                len(demographics_filtered),
                help="Number of block groups in current filter"
            )
        
        with col2:
            total_k12 = int(demographics_filtered['k12_pop'].sum())
            st.metric(
                "Total K-12 Students", 
                f"{total_k12:,}",
                help="Total K-12 population in filtered area"
            )
        
        with col3:
            # Only calculate average from valid (non-NaN) income values
            valid_incomes = demographics_filtered['income'].dropna()
            if len(valid_incomes) > 0:
                avg_income = int(valid_incomes.mean())
                st.metric(
                    "Avg Household Income", 
                    f"${avg_income:,}",
                    help="Average median household income (excludes block groups with no data)"
                )
            else:
                st.metric(
                    "Avg Household Income", 
                    "No data",
                    help="No valid income data available"
                )
        
        with col4:
            if 'marketing_priority' in demographics_filtered.columns:
                high_priority = len(demographics_filtered[demographics_filtered['marketing_priority'] >= 6])
                st.metric(
                    "High Priority Areas", 
                    high_priority,
                    help="Block groups with marketing priority ≥6"
                )
        
        # Main map
        st.subheader(f"📍 {selected_metric} by Block Group")
        
        if color_options[selected_metric] in demographics_filtered.columns:
            fig = create_choropleth_map(
                gdf_filtered, 
                demographics_filtered, 
                color_options[selected_metric],
                f"{selected_metric} across Philadelphia Block Groups",
                show_current_students,
                current_students if show_current_students else None
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.error(f"Data not available for {selected_metric}")
        
        # Analysis tables
        st.subheader("📊 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
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
        st.warning("🔍 No block groups match your current filters. Try expanding your criteria.")
        
        # Show available ranges
        if not demographics.empty:
            st.info("**Available Data Ranges:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"• Income: ${demographics['income'].min():,.0f} - ${demographics['income'].max():,.0f}")
                st.write(f"• K-12 Population: {demographics['k12_pop'].min():.0f} - {demographics['k12_pop'].max():.0f}")
            with col2:
                st.write(f"• Total Block Groups: {len(demographics)}")
                st.write(f"• Distance Range: 0 - 35 km from CCA campuses")

if __name__ == "__main__":
    main()