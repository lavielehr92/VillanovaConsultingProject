import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from educational_desert_index import compute_edi, haversine_km
import requests
import numpy as np

# Page config
st.set_page_config(page_title="Philadelphia Educational Desert Explorer", layout="wide", initial_sidebar_state="expanded")

# Load data
demographics = pd.read_csv('demographics.csv')
schools = pd.read_csv('schools.csv')
current_students = pd.read_csv('current_students.csv')

# CCA Campus locations (corrected coordinates)
cca_campuses = pd.DataFrame({
    'name': ['CCA Main Campus (58th St)', 'CCA Baltimore Ave Campus'],
    'lat': [39.9386, 39.9508],  # Corrected Baltimore Ave coordinates
    'lon': [-75.2312, -75.2085],  # Corrected Baltimore Ave coordinates
    'address': ['1939 S. 58th St. Philadelphia, PA', '4109 Baltimore Ave Philadelphia, PA'],
    'type': 'CCA Campus'
})

# Prepare data for EDI
bg_df = demographics.rename(columns={'ZIP': 'geoid_bg'})
schools_df = schools.rename(columns={'school_name': 'school_id'})

# Compute EDI
edi_df = compute_edi(bg_df, schools_df)
edi_df['geoid_bg'] = edi_df['geoid_bg'].astype(int)

# Add proximity analysis - distance to nearest CCA campus
def calculate_proximity_score(zip_lat, zip_lon):
    """Calculate proximity score based on distance to nearest CCA campus"""
    distances = []
    for _, campus in cca_campuses.iterrows():
        dist = haversine_km(zip_lat, zip_lon, campus['lat'], campus['lon'])
        distances.append(dist)
    min_distance = min(distances)
    # Convert to score: closer = higher score (max 5 points for < 1km, decreasing)
    if min_distance < 1:
        return 5
    elif min_distance < 3:
        return 4
    elif min_distance < 5:
        return 3
    elif min_distance < 10:
        return 2
    elif min_distance < 15:
        return 1
    else:
        return 0

# Add proximity scores to demographics
demographics['distance_to_cca'] = demographics.apply(
    lambda row: min([haversine_km(row['lat'], row['lon'], campus['lat'], campus['lon']) 
                    for _, campus in cca_campuses.iterrows()]), axis=1)
demographics['proximity_score'] = demographics.apply(
    lambda row: calculate_proximity_score(row['lat'], row['lon']), axis=1)

# Load current student addresses
import numpy as np

# CCA School locations
cca_schools = pd.DataFrame({
    'name': ['Cornerstone Christian Academy - Main Campus', 'Cornerstone Christian Academy - Baltimore Campus'],
    'address': ['1939 S. 58th St. Philadelphia, PA', '4109 Baltimore Ave Philadelphia, PA'],
    'lat': [39.9386, 39.9508],
    'lon': [-75.2312, -75.2085],
    'type': ['CCA Campus', 'CCA Campus']
})

# Load current student addresses from CSV
try:
    student_addresses = pd.read_csv('Cornerstone/CCA addresses.csv')
    # Add approximate coordinates for student addresses (simplified for demo)
    # In real implementation, you'd geocode these addresses
    np.random.seed(42)  # For consistent demo data
    student_addresses['lat'] = 39.93 + np.random.normal(0, 0.02, len(student_addresses))
    student_addresses['lon'] = -75.23 + np.random.normal(0, 0.02, len(student_addresses))
    student_addresses['type'] = 'Current Student'
except:
    student_addresses = pd.DataFrame(columns=['Street Address', 'lat', 'lon', 'type'])

# Calculate distances from CCA main campus
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3959  # miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Add distance from main CCA campus to demographics
main_campus_lat, main_campus_lon = 39.9386, -75.2312
demographics['distance_to_cca'] = haversine_distance(
    demographics['lat'], demographics['lon'], 
    main_campus_lat, main_campus_lon
)

# Sidebar for controls
with st.sidebar:
    st.header("🎯 CCA Expansion Analysis")
    
    # Distance from School Filter
    st.subheader("📍 Proximity Targeting")
    max_distance = st.slider("Max Distance from CCA (km)", 
                            min_value=1.0, max_value=20.0, 
                            value=8.0, step=0.5, key="distance_slider",
                            help="Students closer to campus have higher enrollment rates")
    
    # Map Overlays
    st.subheader("🗺️ Map Overlays")
    show_campuses = st.checkbox("Show CCA Campuses", value=True, key="show_campuses")
    show_student_overlay = st.checkbox("Show Current Students", value=True, key="show_student_overlay",
                                      help="Display current student home locations")
    show_distance_rings = st.checkbox("Show Distance Rings", value=False, key="distance_rings")
    
    # EDI Range Filter  
    st.subheader("🏫 Educational Access")
    edi_range = st.slider("Educational Desert Index Range", 
                         min_value=float(edi_df['edi'].min()), 
                         max_value=float(edi_df['edi'].max()), 
                         value=(float(edi_df['edi'].min()), float(edi_df['edi'].max())),
                         step=1.0, key="edi_slider")
    
    # Income targeting with presets
    st.subheader("💰 Income Targeting")
    income_preset = st.selectbox("Income Target Preset", 
                                ["All Incomes", "Premium Tier ($200K+)", "High Income ($100K+)", 
                                 "Upper Middle ($75K+)", "Middle Class ($50K+)", "Custom Range"],
                                key="income_preset")
    
    if income_preset == "Custom Range":
        target_income_range = st.slider("Custom Income Range", 
                                       min_value=0, max_value=350000, 
                                       value=(40000, 350000), step=5000,
                                       format="$%d", key="custom_income")
    elif income_preset == "Premium Tier ($200K+)":
        target_income_range = (200000, 350000)
        st.info("🏆 **Premium**: $200K+ (Full tuition + major donors)")
    elif income_preset == "High Income ($100K+)":
        target_income_range = (100000, 350000)
        st.info("💎 **High Income**: $100K+ (Tiered tuition eligible)")
    elif income_preset == "Upper Middle ($75K+)":
        target_income_range = (75000, 350000)
        st.info("🎯 **Upper Middle**: $75K+ (Moderate assistance)")
    elif income_preset == "Middle Class ($50K+)":
        target_income_range = (50000, 350000)
        st.info("🏡 **Middle Class**: $50K+ (Scholarship eligible)")
    else:  # All Incomes
        target_income_range = (0, 350000)
    
    # Target Family Criteria
    st.subheader("👨‍👩‍👧‍👦 Family Criteria")
    min_first_gen = st.slider("Min % First Generation", 0, 100, 0, key="first_gen_slider")
    min_christian = st.slider("Min % Christian", 0, 100, 0, key="christian_slider")
    
    # Map Display Options
    st.subheader("🗺️ Map Display")
    show_marketing_priority = st.checkbox("Show Marketing Priority", value=True, key="show_marketing")
    show_current_students = st.checkbox("Show Current Student Addresses", value=True, key="show_students")
    show_schools = st.checkbox("Show Competitor Schools", value=True, key="show_schools")
    
    # Target Family Filters
    st.subheader("Target Family Criteria")
    min_first_gen = st.slider("Min % First Generation", 0, 100, 0)
    min_christian = st.slider("Min % Christian", 0, 100, 0)
    
    # Income targeting with presets
    income_preset = st.selectbox("Income Target Preset", 
                                ["Custom Range", "Premium Tier ($150K+)", "High Income ($100K+)", 
                                 "Upper Middle ($75K+)", "Middle Class ($50K+)", "All Incomes"])
    
    if income_preset == "Custom Range":
        target_income_range = st.slider("Target Income Range", 
                                       min_value=0, max_value=350000, 
                                       value=(40000, 350000), step=5000,
                                       format="$%d")
    elif income_preset == "Premium Tier ($150K+)":
        target_income_range = (150000, 350000)
        st.write("🏆 **Premium Tier**: $150K - $350K (Full tuition + donations)")
    elif income_preset == "High Income ($100K+)":
        target_income_range = (100000, 350000)
        st.write("💎 **High Income**: $100K - $350K (Tiered tuition eligible)")
    elif income_preset == "Upper Middle ($75K+)":
        target_income_range = (75000, 350000)
        st.write("🎯 **Upper Middle**: $75K - $350K (Moderate assistance)")
    elif income_preset == "Middle Class ($50K+)":
        target_income_range = (50000, 350000)
        st.write("🏡 **Middle Class**: $50K - $350K (Scholarship eligible)")
    else:  # All Incomes
        target_income_range = (0, 350000)
        st.write("🌍 **All Families**: Full income spectrum")
    
    # Marketing Priority Toggle
    show_marketing_priority = st.checkbox("Highlight Marketing Priority Areas", value=True)
    
    # School Type Filter
    school_types = st.multiselect("School Types", 
                                 schools['type'].unique(), 
                                 default=schools['type'].unique())
    
    # Poverty Rate Filter
    poverty_range = st.slider("Poverty Rate Range (%)", 
                             min_value=float(demographics['poverty_rate'].min()), 
                             max_value=float(demographics['poverty_rate'].max()), 
                             value=(float(demographics['poverty_rate'].min()), float(demographics['poverty_rate'].max())),
                             step=0.5)
    
    st.markdown("---")
    st.header("Block Details")
    
    # ZIP Selection for details
    selected_zip = st.selectbox("Select ZIP Code for Details", 
                               sorted(demographics['ZIP'].unique()),
                               key="zip_selector")
    
    if selected_zip:
        zip_edi = edi_df[edi_df['geoid_bg'] == selected_zip]
        zip_demo = demographics[demographics['ZIP'] == selected_zip]
        
        if not zip_edi.empty and not zip_demo.empty:
            st.subheader(f"ZIP {selected_zip}")
            
            # EDI Score with color coding
            edi_score = zip_edi['edi'].values[0]
            if edi_score > 60:
                st.error(f"🚨 High Desert Risk: {edi_score:.1f}")
            elif edi_score > 40:
                st.warning(f"⚠️ Moderate Risk: {edi_score:.1f}")
            else:
                st.success(f"✅ Good Access: {edi_score:.1f}")
            
            st.metric("Median Income", f"${zip_demo['income'].values[0]:,}")
            st.metric("K-12 Population", f"{zip_edi['k12_pop'].values[0]:,.0f}")
            st.metric("Poverty Rate", f"{zip_demo['poverty_rate'].values[0]:.1f}%")
            st.metric("% First Generation", f"{zip_demo['%first_gen'].values[0]:.1f}%")
            st.metric("% Christian", f"{zip_demo['%Christian'].values[0]:.1f}%")
            st.metric("Nearest School", f"{zip_edi['nearest_km'].values[0]:.2f} km")
            st.metric("Distance to CCA", f"{zip_demo['distance_to_cca'].values[0]:.2f} km")
            
            # Proximity assessment
            distance = zip_demo['distance_to_cca'].values[0]
            if distance <= 2:
                st.success("🎯 **OPTIMAL DISTANCE**: Very close to campus")
            elif distance <= 5:
                st.info("✅ **GOOD DISTANCE**: Close to campus") 
            elif distance <= 8:
                st.warning("⚠️ **MODERATE DISTANCE**: Reasonable commute")
            else:
                st.error("❌ **FAR DISTANCE**: Long commute may reduce enrollment")
            
            # Marketing Assessment
            st.markdown("**Marketing Assessment:**")
            income = zip_demo['income'].values[0]
            first_gen = zip_demo['%first_gen'].values[0]
            christian = zip_demo['%Christian'].values[0]
            
            # Calculate marketing score
            marketing_score = 0
            if income >= 60000:
                marketing_score += 2
            if first_gen >= 30:
                marketing_score += 2
            if christian >= 25:
                marketing_score += 1
            if edi_score >= 40:
                marketing_score += 1
            
            # Display marketing priority with enhanced tiers
            if marketing_score >= 8:
                st.success("� **ULTRA HIGH PRIORITY**: Premium donor + enrollment target")
                st.write("✨ $200K+ income + High first-gen + Strong community")
            elif marketing_score >= 6:
                st.success("💎 **TIER 1 PRIORITY**: Prime target for CCA marketing")
                st.write("🎯 $150K+ income + First-gen families + Faith community")
            elif marketing_score >= 4:
                st.warning("🔥 **TIER 2 TARGET**: Strong marketing opportunity")
                reasons = []
                if income >= 100000:
                    reasons.append("High income ($100K+)")
                elif income >= 75000:
                    reasons.append("Upper income ($75K+)")
                if first_gen >= 35:
                    reasons.append("High first-generation")
                elif first_gen >= 20:
                    reasons.append("Moderate first-generation")
                if christian >= 30:
                    reasons.append("Strong Christian community")
                elif christian >= 20:
                    reasons.append("Christian presence")
                if edi_score >= 40:
                    reasons.append("Underserved area")
                st.write(f"• {' + '.join(reasons)}")
            elif marketing_score >= 2:
                st.info("📋 **TIER 3 CONSIDER**: Secondary marketing target")
            else:
                st.error("❌ **LOW PRIORITY**: Limited marketing potential")
            
            # Specific marketing recommendations
            st.markdown("**Marketing Recommendations:**")
            if income >= 200000 and first_gen >= 50:
                st.write("🏆 **PREMIUM TIER**: $200K+ income - Full tuition + major donor potential")
            elif income >= 150000 and first_gen >= 40:
                st.write("💎 **PLATINUM TIER**: $150K+ income - Full tuition + donation asks")
            elif income >= 100000 and first_gen >= 30:
                st.write("💰 **GOLD TIER**: $100K+ income - Tiered tuition strategy")
            elif income >= 75000:
                st.write("🥉 **SILVER TIER**: $75K+ income - Moderate tuition assistance")
            if christian >= 40:
                st.write("⛪ **Faith-Based Outreach**: Partner with local churches")
            if edi_score >= 60:
                st.write("🏫 **Need-Based Messaging**: Emphasize filling educational gap")
            if income >= 50000 and income < 80000:
                st.write("🎓 **Scholarship Messaging**: Highlight tuition assistance")

# Main content area
st.title("Philadelphia Educational Desert Explorer")
st.markdown("*Identifying underserved areas and expansion opportunities for Cornerstone Christian Academy*")

# Filter EDI data by distance and other criteria
edi_with_distance = edi_df.merge(demographics[['ZIP', 'distance_to_cca']], left_on='geoid_bg', right_on='ZIP', how='left')

filtered_edi = edi_with_distance[
    (edi_with_distance['edi'] >= edi_range[0]) & 
    (edi_with_distance['edi'] <= edi_range[1]) &
    (edi_with_distance['distance_to_cca'] <= max_distance)
]

# Apply distance and income filters
filtered_demo = demographics[
    (demographics['distance_to_cca'] <= max_distance) &
    (demographics['%first_gen'] >= min_first_gen) &
    (demographics['%Christian'] >= min_christian) &
    (demographics['income'] >= target_income_range[0]) &
    (demographics['income'] <= target_income_range[1])
    ]
    
    # Create marketing priority scoring with enhanced tiers including proximity
    demographics_copy = demographics.copy()
    demographics_copy['marketing_score'] = 0
    
    # Proximity scoring (NEW - highest weight for distance)
    demographics_copy.loc[demographics_copy['distance_to_cca'] <= 2, 'marketing_score'] += 4  # Very close
    demographics_copy.loc[(demographics_copy['distance_to_cca'] > 2) & (demographics_copy['distance_to_cca'] <= 5), 'marketing_score'] += 3  # Close
    demographics_copy.loc[(demographics_copy['distance_to_cca'] > 5) & (demographics_copy['distance_to_cca'] <= 8), 'marketing_score'] += 2  # Moderate
    demographics_copy.loc[(demographics_copy['distance_to_cca'] > 8) & (demographics_copy['distance_to_cca'] <= 12), 'marketing_score'] += 1  # Far
    
    # Income scoring (higher weight for premium incomes)
    demographics_copy.loc[demographics_copy['income'] >= 200000, 'marketing_score'] += 5  # Premium
    demographics_copy.loc[(demographics_copy['income'] >= 150000) & (demographics_copy['income'] < 200000), 'marketing_score'] += 4  # Platinum
    demographics_copy.loc[(demographics_copy['income'] >= 100000) & (demographics_copy['income'] < 150000), 'marketing_score'] += 3  # Gold
    demographics_copy.loc[(demographics_copy['income'] >= 75000) & (demographics_copy['income'] < 100000), 'marketing_score'] += 2  # Silver
    demographics_copy.loc[(demographics_copy['income'] >= 50000) & (demographics_copy['income'] < 75000), 'marketing_score'] += 1  # Bronze
    
    # First generation scoring (enhanced for higher percentages)
    demographics_copy.loc[demographics_copy['%first_gen'] >= 50, 'marketing_score'] += 3  # Very high
    demographics_copy.loc[(demographics_copy['%first_gen'] >= 35) & (demographics_copy['%first_gen'] < 50), 'marketing_score'] += 2  # High
    demographics_copy.loc[(demographics_copy['%first_gen'] >= 20) & (demographics_copy['%first_gen'] < 35), 'marketing_score'] += 1  # Moderate
    
    # Christian community scoring
    demographics_copy.loc[demographics_copy['%Christian'] >= 30, 'marketing_score'] += 2  # Strong faith
    demographics_copy.loc[(demographics_copy['%Christian'] >= 20) & (demographics_copy['%Christian'] < 30), 'marketing_score'] += 1  # Moderate faith
    
    # Add EDI scores to demographics for marketing calculation
    demo_with_edi = demographics_copy.merge(edi_df[['geoid_bg', 'edi']], left_on='ZIP', right_on='geoid_bg', how='left')
    demo_with_edi.loc[demo_with_edi['edi'] >= 40, 'marketing_score'] += 1  # Underserved
    
    # Add marketing priority to EDI data
    edi_marketing = edi_df.merge(demo_with_edi[['ZIP', 'marketing_score', '%first_gen', '%Christian', 'distance_to_cca']], 
                                left_on='geoid_bg', right_on='ZIP', how='left')
    
    filtered_schools = schools  # Show all schools for context
    
    # Choose what to display based on marketing priority toggle
    if show_marketing_priority:
        map_data = edi_marketing
        color_col = 'marketing_score'
        color_scale = 'Viridis'
        title = 'Marketing Priority Map - Darker = Higher Priority for CCA Outreach'
        labels = {'marketing_score': 'Marketing Priority'}
        hover_data = ['edi', 'marketing_score', '%first_gen', '%Christian', 'nearest_km', 'poverty_rate', 'k12_pop']
    else:
        map_data = filtered_edi
        color_col = 'edi'
        color_scale = 'RdYlBu_r'
        title = 'Educational Desert Index Map - Click bubbles for details'
        labels = {'edi': 'EDI Score'}
        hover_data = ['edi', 'nearest_km', 'poverty_rate', 'k12_pop']
    
    # Filter by distance - handle missing column gracefully
    if 'distance_to_cca' in map_data.columns:
        map_data_filtered = map_data[map_data['distance_to_cca'] <= max_distance]
    else:
        map_data_filtered = map_data
    
    # Main Map
    fig_edi_map = px.scatter_map(map_data_filtered, lat='lat', lon='lon', color=color_col, size='edi',
                                 hover_name='geoid_bg', 
                                 hover_data=hover_data + ['distance_to_cca'],
                                 map_style='open-street-map', zoom=11, 
                                 title=title,
                                 color_continuous_scale=color_scale,
                                 labels=labels,
                                 size_max=25)
    
    # Add CCA school locations
    for _, school in cca_schools.iterrows():
        fig_edi_map.add_trace(go.Scattermap(
            lat=[school['lat']],
            lon=[school['lon']],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            text=school['name'],
            name='CCA Campus',
            showlegend=True
        ))
    
    # Add distance rings if enabled
    if show_distance_rings:
        for radius in [1, 3, 5]:  # miles
            if radius <= max_distance:
                circle_lat, circle_lon = [], []
                for angle in range(0, 361, 5):
                    # Convert radius from miles to degrees (approximate)
                    lat_offset = radius / 69.0  # 1 degree lat ≈ 69 miles
                    lon_offset = radius / (69.0 * np.cos(np.radians(main_campus_lat)))
                    
                    circle_lat.append(main_campus_lat + lat_offset * np.cos(np.radians(angle)))
                    circle_lon.append(main_campus_lon + lon_offset * np.sin(np.radians(angle)))
                
                fig_edi_map.add_trace(go.Scattermap(
                    lat=circle_lat,
                    lon=circle_lon,
                    mode='lines',
                    line=dict(width=1, color='red'),
                    name=f'{radius} mile radius',
                    showlegend=True
                ))
    
    # Add competitor schools if enabled
    if show_schools:
        fig_schools_layer = px.scatter_map(filtered_schools, lat='lat', lon='lon', 
                                          hover_name='school_name',
                                          hover_data=['type', 'tuition', 'rating'], 
                                          color='type',
                                          map_style='open-street-map', zoom=10)
        
        for trace in fig_schools_layer.data:
            trace.marker.symbol = 'diamond'
            trace.marker.size = 10
            fig_edi_map.add_trace(trace)
    
    # Add CCA Campuses with yellow stars
    if show_campuses:
        for _, campus in cca_campuses.iterrows():
            fig_edi_map.add_trace(go.Scattermap(
                lat=[campus['lat']],
                lon=[campus['lon']],
                mode='markers',
                marker=dict(size=25, color='gold', symbol='star', line=dict(color='orange', width=2)),
                text=[campus['name']],
                hovertemplate=f"<b>{campus['name']}</b><br>{campus['address']}<extra></extra>",
                name=campus['name']
            ))
    
    # Add Current Students from CSV
    if show_student_overlay:
        fig_edi_map.add_trace(go.Scattermap(
            lat=current_students['lat'],
            lon=current_students['lon'],
            mode='markers',
            marker=dict(size=8, color='orange', symbol='circle'),
            text=current_students['address'],
            hovertemplate="<b>Current Student</b><br>%{text}<extra></extra>",
            name="Current Students"
        ))
    
    # Add current student addresses if enabled (legacy)
    if show_current_students and not student_addresses.empty:
        fig_edi_map.add_trace(go.Scattermap(
            lat=student_addresses['lat'],
            lon=student_addresses['lon'],
            mode='markers',
            marker=dict(size=6, color='lightblue', symbol='circle-dot'),
            text=student_addresses['Street Address'],
            name='Current Students',
            showlegend=True
        ))
    
    fig_edi_map.update_traces(marker=dict(sizemin=8))
    fig_edi_map.update_layout(
        height=700, 
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(l=0, r=150, t=50, b=0)  # Add right margin for legend
    )
    
    st.plotly_chart(fig_edi_map, key="main_map", use_container_width=True)
    
    # Marketing Summary Statistics
    st.markdown("### Marketing Intelligence Summary")
    col_a, col_b, col_c, col_d = st.columns(4)
    
    ultra_high_areas = len(demo_with_edi[demo_with_edi['marketing_score'] >= 8])
    tier1_areas = len(demo_with_edi[demo_with_edi['marketing_score'] >= 6])
    tier2_areas = len(demo_with_edi[demo_with_edi['marketing_score'] >= 4])
    premium_firstgen = len(demo_with_edi[(demo_with_edi['income'] >= 150000) & (demo_with_edi['%first_gen'] >= 40)])
    high_income_firstgen = len(demo_with_edi[(demo_with_edi['income'] >= 100000) & (demo_with_edi['%first_gen'] >= 30)])
    
    with col_a:
        st.metric("🏆 Ultra High Priority", ultra_high_areas, help="$200K+ Premium targets")
    with col_b:
        st.metric("💎 Tier 1 Priority", tier1_areas, help="$150K+ Prime targets")
    with col_c:
        st.metric("🔥 Tier 2 Targets", tier2_areas, help="$100K+ Strong opportunities")
    with col_d:
        st.metric("💰 Premium First-Gen", premium_firstgen, help="$150K+ with 40%+ first-gen")
    
    # Additional metrics row
    col_e, col_f, col_g, col_h = st.columns(4)
    with col_e:
        st.metric("🎯 High-Income First-Gen", high_income_firstgen, help="$100K+ with 30%+ first-gen")
    with col_f:
        st.metric("🏫 Schools Visible", len(filtered_schools))
    with col_g:
        avg_income = demo_with_edi['income'].mean()
        st.metric("📊 Avg Income", f"${avg_income:,.0f}")
    with col_h:
        max_income = demo_with_edi['income'].max()
        st.metric("💎 Max Income", f"${max_income:,.0f}")
    
    # Target Family Analysis
    st.markdown("### Target Family Insights")
    target_families = demographics[
        (demographics['income'] >= target_income_range[0]) &
        (demographics['income'] <= target_income_range[1]) &
        (demographics['%first_gen'] >= min_first_gen) &
        (demographics['%Christian'] >= min_christian)
    ]
    
    if not target_families.empty:
        st.write(f"**{len(target_families)} ZIP codes** match your target criteria:")
        for _, row in target_families.iterrows():
            zip_code = row['ZIP']
            income = row['income']
            first_gen = row['%first_gen']
            christian = row['%Christian']
            st.write(f"• ZIP {zip_code}: ${income:,} income, {first_gen}% first-gen, {christian}% Christian")
    else:
        st.write("No areas match current target criteria. Adjust filters to find opportunities.")