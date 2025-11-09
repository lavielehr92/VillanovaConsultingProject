# Philadelphia Educational Desert Explorer - Quick Reference

## 📦 Files for GitHub

### Core Application Files
- `app_block_groups.py` - Main Streamlit dashboard
- `educational_desert_index_bg.py` - EDI calculation engine
- `fetch_block_groups_live.py` - Live Census data fetcher
- `fetch_enhanced_k12_data.py` - Enhanced K-12 population estimator

### Configuration Files
- `requirements.txt` - Python dependencies
- `.gitignore` - Git exclusions
- `.env.example` - Environment variable template
- `.streamlit/secrets.toml.example` - Streamlit secrets template

### Documentation
- `README.md` - Main documentation and setup guide
- `METHODOLOGY.md` - Detailed EDI and methodology explanation
- `DEPLOYMENT.md` - Step-by-step deployment guide

### Data Files (Optional - can regenerate)
- `philadelphia_block_groups.geojson` - Block group boundaries
- `demographics_block_groups.csv` - Demographic metrics
- `schools.csv` - School locations

## 🔑 Census API Key Setup

### For Local Development

**Option 1: .env file (Recommended)**
1. Copy `.env.example` to `MyKeys/.env`
2. Add your key: `CENSUS_API_KEY=your_key_here`

**Option 2: Environment variable**
```bash
export CENSUS_API_KEY=your_key_here  # Linux/Mac
set CENSUS_API_KEY=your_key_here     # Windows CMD
$env:CENSUS_API_KEY="your_key_here"  # Windows PowerShell
```

### For Streamlit Cloud Deployment

1. Deploy app to Streamlit Cloud
2. Go to app settings
3. Add secret:
```toml
CENSUS_API_KEY = "your_key_here"
```

## 📊 EDI Methodology Summary

### Educational Desert Index (EDI)

**Score Range**: 0-100 (higher = more underserved)

**Components**:
1. **School Accessibility (30%)** - Two-Step Floating Catchment Area
   - Measures supply/demand ratio
   - Accounts for school capacity

2. **Gravity Access (25%)** - Distance-weighted accessibility
   - Exponential decay (5km half-life)
   - Closer schools weighted more heavily

3. **Geographic Isolation (25%)** - Distance to nearest school
   - Direct barrier measurement
   - Critical for emergency access

4. **Neighborhood Need (20%)** - Socioeconomic factors
   - Poverty rate (70%)
   - Low educational attainment (30%)

**Formula**:
```python
EDI = 30% × (1 - accessibility) +
      25% × (1 - gravity) +
      25% × distance_normalized +
      20% × need_score
```

**Interpretation**:
- **80-100**: Severe desert - immediate priority
- **60-79**: High need - strong intervention recommended
- **40-59**: Moderate need - consider for expansion
- **20-39**: Low need - adequate access
- **0-19**: Well-served - monitor only

## 🎯 Marketing Priority Logic

**Score Range**: 0-10 (higher = better fit for CCA)

### Scoring Components:

**1. Proximity (4 points max - 40% weight)**
```python
distance = min(distance_to_CCA_campuses)
if distance < 2km:    score += 4
elif distance < 5km:  score += 3
elif distance < 10km: score += 2
elif distance < 20km: score += 1
```

**2. Income Alignment (3 points max - 30% weight)**
```python
# Target: Can afford tuition but not wealthy
if $25K < income < $50K:    score += 3  # Sweet spot
elif $50K < income < $75K:  score += 2  # Upper middle
elif $75K < income < $100K: score += 1  # High income
```

**3. Mission Alignment (2 points max - 20% weight)**
```python
# First-generation immigrants
if first_gen_pct >= 40%: score += 2
elif first_gen_pct >= 25%: score += 1

# Christian population
if christian_pct >= 30%: score += 1
```

**4. Market Size (1 point max - 10% weight)**
```python
if k12_population >= 200: score += 1
```

### Priority Tiers:
- **8-10**: Immediate action - ideal fit
- **6-7**: High priority - strong potential
- **4-5**: Moderate - worth consideration
- **0-3**: Lower priority - limited alignment

## 🚀 Quick Start Commands

### Initial Setup
```bash
git clone https://github.com/YOUR_USERNAME/philly-education-desert.git
cd philly-education-desert
pip install -r requirements.txt
```

### Fetch Data
```bash
# Get Census data (requires API key)
python fetch_block_groups_live.py

# Enhance K-12 estimates
python fetch_enhanced_k12_data.py
```

### Run Dashboard
```bash
streamlit run app_block_groups.py
```

## 📈 Data Quality Improvements

### K-12 Population Enhancement

**Problem**: Block group ACS data had high suppression → 81,160 students (too low)

**Solution**: Tract-level disaggregation
1. Fetch tract-level K-12 data (more reliable)
2. Disaggregate to block groups using population ratios
3. Result: 126,809 students (+36% improvement)

**Method**:
```python
bg_k12_pop = tract_k12_pop × (bg_total_pop / tract_total_pop)
```

### Why This Works:
- Tract data has lower margins of error
- Proportional allocation preserves geographic accuracy
- Handles non-residential block groups gracefully

## 🗺️ Understanding the Map

### Color Scale
- **Red/Dark**: High EDI (educational deserts)
- **Yellow/Orange**: Moderate EDI
- **Blue/Light**: Low EDI (well-served)

### Symbols
- **⭐ Gold Stars**: CCA campus locations
- **🔵 Blue Dots**: Current CCA students (if enabled)

### Hover Information
- Block Group ID & Tract
- K-12 Population
- Median Income
- Poverty Rate
- Total Population
- Households with children <18
- Race demographics (% Black, % White)
- Selected metric value

## 🔧 Troubleshooting

### Map Shows 0 Population
1. Clear cache: Click "⋮" menu → "Clear cache"
2. Restart app: `Ctrl+C` then `streamlit run app_block_groups.py`
3. Re-fetch data: `python fetch_enhanced_k12_data.py`

### API Key Not Working
- Verify key at https://api.census.gov/data/key_signup.html
- Check environment variable is set correctly
- Ensure no extra spaces in `.env` file
- Try alternative variable names (CENSUSBUREAUAPI_KEY)

### GDAL/Geospatial Errors
```bash
# Windows
conda install -c conda-forge geopandas

# Linux/Mac
sudo apt-get install gdal-bin libgdal-dev
pip install --upgrade geopandas pyogrio
```

## 📚 Key Metrics Explained

### K-12 Population
- **Source**: Census ACS B01001 (ages 5-17)
- **Enhancement**: Tract-level disaggregated to block groups
- **Total**: ~127K students in Philadelphia
- **Zero values**: Non-residential areas (parks, industrial, water)

### Median Income
- **Source**: Census ACS B19013_001E
- **Per**: Block group median (not individual households)
- **Sentinel**: -666666666 = no data available
- **Range**: ~$11K to ~$200K in Philadelphia

### Poverty Rate
- **Source**: Census ACS B17001_002E / B17001_001E
- **Definition**: % below federal poverty line
- **Calculation**: (pop_below_poverty / total_pop_for_poverty) × 100

### Total Population
- **Source**: Census ACS B01003_001E
- **Purpose**: Denominator for disaggregation
- **Zero blocks**: ~5% (64 of 1,338) are non-residential

## 🌐 Deployment Checklist

- [ ] Get Census API key
- [ ] Push code to GitHub
- [ ] Deploy to Streamlit Cloud
- [ ] Add API key to secrets
- [ ] Test live deployment
- [ ] Verify map loads with data
- [ ] Test filters and interactions
- [ ] Set up data refresh schedule
- [ ] Document for team
- [ ] Share access link

## 📞 Support

**Repository**: https://github.com/YOUR_USERNAME/philly-education-desert  
**Documentation**: See README.md and METHODOLOGY.md  
**Issues**: https://github.com/YOUR_USERNAME/philly-education-desert/issues

## 🎓 Citation

```bibtex
@software{philly_education_desert_2025,
  title = {Philadelphia Educational Desert Explorer},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/philly-education-desert},
  note = {Interactive dashboard for educational access analysis}
}
```
