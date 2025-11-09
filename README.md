# Philadelphia Educational Desert Explorer
## CCA Expansion Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing educational opportunities and identifying expansion opportunities for Cornerstone Christian Academy (CCA) in Philadelphia.

## 📊 Dashboard Features

### Core Functionality
- **Educational Desert Index (EDI)**: Sophisticated algorithm combining multiple access metrics
- **Marketing Priority Analysis**: Data-driven targeting for families earning up to $350K
- **Interactive Maps**: Plotly-powered visualizations with demographic overlays
- **Proximity Analysis**: Distance-based scoring from existing CCA campuses
- **Real-time Filtering**: Dynamic ZIP code analysis based on multiple criteria

### Key Visualizations
- ZIP code-level demographic mapping
- School location overlays (public, private, charter)
- Current student distribution
- Marketing intelligence summary metrics

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit pandas plotly scikit-learn numpy requests
```

### Running the Dashboard
```bash
streamlit run app_fixed.py
```

The dashboard will open at `http://localhost:8501`

## 📁 Project Structure

```
MBA8583/
├── app_fixed.py                    # Main dashboard application
├── educational_desert_index.py     # EDI calculation algorithms
├── demographics.csv                # ZIP code demographic data (Census-sourced)
├── schools.csv                     # School location and type data
├── current_students.csv            # Current CCA student addresses
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## 📈 Data Sources

### Demographics Data (`demographics.csv`)
- **Source**: U.S. Census Bureau ACS 5-Year 2022 data
- **Coverage**: 57 Philadelphia ZIP codes
- **Key Fields**:
  - `k12_pop`: K-12 student population (Census API verified)
  - `income`: Median household income
  - `poverty_rate`: Percentage below poverty line
  - `%Christian`: Estimated Christian population percentage
  - `%first_gen`: First-generation college students percentage
  - `lat/lon`: Geographic coordinates

### Schools Data (`schools.csv`)
- School locations with coordinates
- Type classification (public, private, charter)
- Tuition and rating information
- Capacity estimates

### Current Students (`current_students.csv`)
- Current CCA student home addresses
- Geocoded for proximity analysis

## 🎯 Educational Desert Index (EDI)

The EDI combines multiple accessibility metrics:

1. **Two-Step Floating Catchment Analysis (2SFCA)**: Supply/demand ratio
2. **Gravity-Based Access**: Distance-decay accessibility modeling
3. **Nearest School Distance**: Geographic isolation metric
4. **Neighborhood Need**: Poverty rate and education gaps

**Formula**: Weighted combination scaled 0-100 (higher = more underserved)

## 🎨 Dashboard Controls

### Sidebar Filters
- **Proximity Targeting**: Distance filter from CCA campuses (1-35km)
- **Map Overlays**: Current students toggle
- **Educational Access**: EDI range slider (4.23-75.05)
- **Household Income**: Income range targeting ($35K-$350K)
- **School Types**: Filter visible school types
- **Target Criteria**: First-gen and Christian population filters

### Marketing Priority Algorithm
Scores ZIP codes based on:
- Distance to CCA campuses (highest weight)
- Household income brackets
- First-generation college percentage
- Christian population estimates
- Educational desert severity (EDI scores)

## 🔧 Technical Implementation

### Key Libraries
- **Streamlit**: Web application framework
- **Plotly**: Interactive mapping and visualization
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning preprocessing
- **NumPy**: Numerical computations

### Performance Considerations
- Efficient haversine distance calculations
- Optimized data filtering and merging
- Responsive map rendering with appropriate zoom levels

## 📊 Marketing Intelligence Metrics

The dashboard provides actionable metrics:
- **Ultra High Priority**: Areas with marketing scores ≥8
- **Tier 1 Priority**: Areas with scores ≥6
- **Tier 2 Targets**: Areas with scores ≥4
- **Premium First-Gen**: High-income + high first-generation families

## 🎯 Use Cases

1. **Expansion Planning**: Identify underserved areas for new campuses
2. **Marketing Strategy**: Target high-potential ZIP codes for enrollment
3. **Resource Allocation**: Understand current student distribution
4. **Competitive Analysis**: Visualize school density and gaps
5. **Demographic Research**: Explore Philadelphia education landscape

## 🛠 Customization

### Adding New Data Sources
1. Update CSV files with new data
2. Modify column mappings in `app_fixed.py`
3. Adjust EDI weights in `educational_desert_index.py`

### Extending Analysis
- Add new demographic variables
- Implement additional scoring algorithms
- Create custom visualizations
- Export analysis results

## 📝 Notes for Collaborators

### Data Quality
- K-12 population data sourced from official Census API
- Geographic coordinates verified for accuracy
- Income data represents median household estimates

### Known Limitations
- Some ZIP codes may lack complete Census data
- Private school data may have gaps
- Christian population percentages are estimates

### Future Enhancements
- Real-time Census data integration
- Advanced demographic forecasting
- Machine learning predictive models
- Export functionality for analysis results

## 📞 Support

For questions or contributions, contact the development team or submit issues through your preferred collaboration platform.

---
*Built for MBA 8583 - Strategic Analysis*  
*Cornerstone Christian Academy Expansion Planning*

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`

## Data Connection

The app uses placeholder CSV files. To connect real data:

- `demographics.csv`: Replace with data from Philly Open Data or NCES. Columns: ZIP, income, EDI, %Christian
- `schools.csv`: Data from NCES or CCA internal. Columns: school_name, type, tuition, rating, lat, lon
- `outreach_plan.csv`: Internal plan data. Columns: month, key_initiative, channel

For census tract choropleth, integrate geojson from Philly Open Data.

## Features

- Interactive filters
- Maps and charts
- Timeline for outreach