import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from educational_desert_index import compute_edi
import requests

# Load data
demographics = pd.read_csv('demographics.csv')
schools = pd.read_csv('schools.csv')
outreach = pd.read_csv('outreach_plan.csv')

# Prepare data for EDI
bg_df = demographics.rename(columns={'ZIP': 'geoid_bg'})
schools_df = schools.rename(columns={'school_name': 'school_id'})

# Compute EDI
edi_df = compute_edi(bg_df, schools_df)
edi_df['geoid_bg'] = edi_df['geoid_bg'].astype(int)

# Sidebar filters
st.sidebar.header("Filters")
selected_zip = st.sidebar.multiselect("Select ZIP Codes", demographics['ZIP'].unique(), default=demographics['ZIP'].unique())
income_bracket = st.sidebar.slider("Income Bracket", min_value=int(demographics['income'].min()), max_value=int(demographics['income'].max()), value=(int(demographics['income'].min()), int(demographics['income'].max())))
school_type = st.sidebar.multiselect("School Type", schools['type'].unique(), default=schools['type'].unique())

# Filter data
filtered_demo = demographics[demographics['ZIP'].isin(selected_zip) & (demographics['income'] >= income_bracket[0]) & (demographics['income'] <= income_bracket[1])]
filtered_schools = schools[schools['type'].isin(school_type)]

# Interactive Details
st.sidebar.header("Details")
selected_zip_detail = st.sidebar.selectbox("Select ZIP for details", filtered_demo['ZIP'].unique() if not filtered_demo.empty else [])
if selected_zip_detail:
    zip_data = filtered_demo[filtered_demo['ZIP'] == selected_zip_detail]
    st.sidebar.write(f"**ZIP {selected_zip_detail} Details:**")
    st.sidebar.write(f"Median Income: ${zip_data['income'].values[0]}")
    st.sidebar.write(f"EDI: {zip_data['EDI'].values[0]}")
    st.sidebar.write(f"% Christian: {zip_data['%Christian'].values[0]}%")
    st.sidebar.write(f"% First Gen: {zip_data['%first_gen'].values[0]}%")

# Overview Section
st.header("Overview / Mission Alignment")
st.write("**CCA Mission:** To provide Christ-centered education in underserved areas.")
col1, col2, col3 = st.columns(3)
col1.metric("Total Enrollment", "150")
col2.metric("Tuition Support Provided", "$200,000")
col3.metric("Average Tuition Assistance", "75%")

# Educational Desert Map
st.header("Educational Desert Map")
st.write("Computed Educational Desert Index (EDI) by ZIP using supply-demand access, proximity, and need.")

# Horizontal bar chart for EDI
fig_map = px.bar(edi_df, x='edi', y='geoid_bg', orientation='h', color='edi', title='Educational Desert Index by ZIP')
st.plotly_chart(fig_map, key="edi_chart")

# Interactive EDI Map with bubbles
fig_edi_map = px.scatter_map(edi_df, lat='lat', lon='lon', color='edi', size='edi',
                             hover_name='geoid_bg', 
                             hover_data=['edi', 'nearest_km', 'poverty_rate', 'k12_pop'],
                             map_style='open-street-map', zoom=10, 
                             title='Educational Desert Index Map',
                             color_continuous_scale='RdYlBu_r',
                             labels={'edi': 'EDI Score'},
                             size_max=25)
fig_edi_map.update_traces(marker=dict(sizemin=8))
fig_edi_map.update_layout(height=600)
st.plotly_chart(fig_edi_map, key="edi_map", use_container_width=True)

# Schools as layers
st.subheader("Nearby Schools")
fig_schools = px.scatter_map(filtered_schools, lat='lat', lon='lon', hover_name='school_name',
                                hover_data=['type', 'tuition', 'rating'], color='type',
                                map_style='open-street-map', zoom=10)
st.plotly_chart(fig_schools, key="schools_map1")

# Family Market Profiles
st.header("Family Market Profiles")
# Define segments based on income
def get_segment(income):
    if income < 40000:
        return 'under-resourced'
    elif income < 100000:
        return 'middle-income'
    else:
        return 'high-income'

filtered_demo['segment'] = filtered_demo['income'].apply(get_segment)
segment_data = filtered_demo.groupby('segment').agg({
    'income':'mean', 
    '%Christian':'mean', 
    '%first_gen':'mean',
    'ZIP':'count'
}).rename(columns={'ZIP':'population'})

# Separate charts for clarity
col1, col2 = st.columns(2)
with col1:
    fig_pop = px.bar(segment_data, x=segment_data.index, y='population', title='Population by Segment')
    st.plotly_chart(fig_pop, key="pop_chart")
with col2:
    fig_income = px.bar(segment_data, x=segment_data.index, y='income', title='Average Income by Segment')
    st.plotly_chart(fig_income, key="income_chart")

fig_christian = px.bar(segment_data, x=segment_data.index, y='%Christian', title='% Christian by Segment')
st.plotly_chart(fig_christian, key="christian_chart")

fig_firstgen = px.bar(segment_data, x=segment_data.index, y='%first_gen', title='% First Generation by Segment')
st.plotly_chart(fig_firstgen, key="firstgen_chart")

st.info("Target high-income first-generation students (>100K) for tiered tuition model.")

# Competitor Landscape
st.header("Competitor Landscape")
st.subheader("Schools Map")
st.plotly_chart(fig_schools, key="schools_map2")  # Reuse with key

st.subheader("Schools Table")
st.dataframe(filtered_schools[['school_name', 'type', 'tuition', 'rating']])

# Expansion Opportunity View
st.header("Expansion Opportunity View")
potential_students = filtered_demo.groupby('ZIP')['income'].count() * 10  # Placeholder
fig_expansion = px.bar(potential_students, title='Potential Students by ZIP')
st.plotly_chart(fig_expansion, key="expansion_chart")

col1, col2 = st.columns(2)
col1.metric("Potential New Students", str(potential_students.sum()))
col2.metric("Estimated Donor Need", "$500,000")

# Outreach & Marketing Plan
st.header("Outreach & Marketing Plan")
st.write("12-Month Plan")
for _, row in outreach.iterrows():
    st.write(f"**{row['month']}:** {row['key_initiative']} via {row['channel']}")

# Insight boxes
st.info("High need, low access zone west of 48th Street.")
st.success("Strong outreach opportunity in University City.")