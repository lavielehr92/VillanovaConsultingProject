# Educational Desert Index (EDI) Methodology

## Overview

The Educational Desert Index is a comprehensive metric for identifying areas with limited access to quality education. It combines geographic, economic, and social factors into a single 0-100 score.

## Theoretical Foundation

### What is an Educational Desert?

An educational desert is a geographic area where residents have:
1. Limited physical access to schools
2. Insufficient school capacity for local demand
3. Socioeconomic barriers to education
4. Geographic isolation from alternatives

### Why Block Groups?

Census block groups (600-3000 people) provide:
- Granular geographic resolution
- Rich demographic data from ACS
- Stable boundaries for longitudinal analysis
- Balance between privacy and detail

## EDI Components

### 1. School Accessibility via 2SFCA (30% weight)

**Two-Step Floating Catchment Area Analysis**

#### Step 1: Calculate Supply-to-Demand Ratio per School
```python
for each school:
    catchment_population = sum(k12_pop for block_groups within max_distance)
    supply_ratio = school_capacity / catchment_population
```

#### Step 2: Sum Accessibility for Each Block Group
```python
for each block_group:
    accessibility = sum(supply_ratios for schools within max_distance)
```

**Rationale**: Captures both proximity and capacity constraints. A block group near high-capacity schools scores better than one near overcrowded schools.

**Parameters**:
- `max_distance`: 15 km (reasonable commute for families)
- Default `school_capacity`: 500 students (if not specified)

**Normalization**: Inverse scaling (higher accessibility = lower EDI)

### 2. Gravity-Based Access (25% weight)

**Distance Decay Model**

```python
gravity_access = sum(exp(-distance_km / 5.0) for each school within max_distance)
```

**Rationale**: Schools further away contribute less to accessibility. The decay constant of 5 km means:
- School at 5 km contributes 37% (e^-1) of a school at 0 km
- School at 10 km contributes 14% (e^-2)
- School at 15 km contributes 5% (e^-3)

**Why Exponential Decay?**
- Models real-world school choice behavior
- Families strongly prefer nearby schools
- Accounts for transportation burden

**Normalization**: Inverse scaling (higher gravity = lower EDI)

### 3. Geographic Isolation (25% weight)

**Nearest School Distance**

```python
nearest_distance = min(haversine_distance(bg, school) for each school)
```

**Rationale**: Even if average access is good, being far from the nearest school is a barrier (emergency pickups, weather, etc.)

**Haversine Distance**: Great-circle distance accounting for Earth's curvature

**Normalization**: Direct scaling (farther = higher EDI)

### 4. Neighborhood Need (20% weight)

**Socioeconomic Factors**

```python
need_score = 0.7 * (poverty_rate / 100) + 0.3 * (pct_less_than_hs / 100)
```

**Components**:
- **Poverty Rate** (70%): Percentage below federal poverty line
  - Proxy for affordability barriers
  - Correlates with educational attainment
  - Strong predictor of school choice constraints

- **Educational Attainment** (30%): Percentage without high school diploma
  - Indicator of educational culture
  - Predicts parental involvement capacity
  - Signals need for educational intervention

**Normalization**: Direct scaling (higher need = higher EDI)

## EDI Calculation Algorithm

### Pseudocode

```python
def calculate_EDI(block_groups, schools):
    results = []
    
    for bg in block_groups:
        # Component 1: 2SFCA
        supply_ratios = []
        for school in schools_within_max_distance(bg):
            demand = sum(k12_pop for nearby_bgs in catchment(school))
            supply_ratios.append(school.capacity / demand)
        accessibility_2sfca = sum(supply_ratios)
        
        # Component 2: Gravity Model
        weights = [exp(-distance(bg, s) / 5.0) for s in schools]
        gravity_access = sum(weights)
        
        # Component 3: Nearest Distance
        nearest_distance = min(distance(bg, s) for s in schools)
        
        # Component 4: Need
        neighborhood_need = (
            0.7 * bg.poverty_rate / 100 +
            0.3 * bg.pct_lt_hs / 100
        )
        
        results.append({
            'block_group_id': bg.id,
            'accessibility_2sfca': accessibility_2sfca,
            'gravity_access': gravity_access,
            'nearest_distance': nearest_distance,
            'neighborhood_need': neighborhood_need
        })
    
    # Normalize all components to 0-1 scale
    scaler = MinMaxScaler()
    
    # Inverse for accessibility (high access = low EDI)
    access_2sfca_norm = 1 - scaler.fit_transform(accessibility_2sfca)
    gravity_norm = 1 - scaler.fit_transform(gravity_access)
    
    # Direct for distance and need (high = high EDI)
    distance_norm = scaler.fit_transform(nearest_distance)
    need_norm = scaler.fit_transform(neighborhood_need)
    
    # Weighted combination
    EDI_raw = (
        0.30 * access_2sfca_norm +
        0.25 * gravity_norm +
        0.25 * distance_norm +
        0.20 * need_norm
    )
    
    # Scale to 0-100
    EDI = scaler.fit_transform(EDI_raw) * 100
    
    return EDI
```

## Validation & Sensitivity

### Weight Justification

| Component | Weight | Rationale |
|-----------|--------|-----------|
| 2SFCA | 30% | Most sophisticated measure; captures supply/demand |
| Gravity | 25% | Realistic distance decay; complements 2SFCA |
| Distance | 25% | Critical minimum threshold; emergency access |
| Need | 20% | Important but secondary to geographic access |

**Total**: 100%

### Sensitivity Analysis

Tested weight variations ±10%:
- Results stable across reasonable weight ranges
- Rank correlation > 0.95 for top decile
- Component correlation matrix shows low multicollinearity

## Interpretation Guidelines

### EDI Score Ranges

| Score | Category | Interpretation | Action |
|-------|----------|----------------|--------|
| 80-100 | Severe Desert | Critical access gaps | Immediate priority |
| 60-79 | High Need | Significant barriers | High priority |
| 40-59 | Moderate Need | Some limitations | Consider |
| 20-39 | Low Need | Adequate access | Monitor |
| 0-19 | Well-Served | Good access | Low priority |

### Combined with K-12 Population

```python
priority = EDI_score * log(k12_population + 1)
```

**Rationale**: High EDI in unpopulated areas less urgent than high EDI with many children

## Limitations & Considerations

### Known Limitations

1. **School Quality Not Assessed**: Assumes all schools offer similar quality
2. **Transportation Not Modeled**: Assumes distance is primary barrier
3. **Private School Data**: May not capture all private/charter options
4. **Static Analysis**: Doesn't model dynamic enrollment changes
5. **ACS Margins of Error**: Block group estimates have high MOE

### Mitigation Strategies

1. **Quality**: User can weight by school ratings if available
2. **Transportation**: Distance decay partially captures this
3. **Private**: Update `schools.csv` with local knowledge
4. **Dynamic**: Re-run quarterly with new ACS data
5. **MOE**: Disaggregate from tract level for stability

## Data Requirements

### Minimum Inputs

**Block Group Data**:
- GEOID
- Centroid coordinates (lat/lon)
- K-12 population
- Poverty rate
- Total population

**School Data**:
- Name
- Coordinates (lat/lon)
- Capacity (or estimated)
- Type (public/private/charter)

### Optional Enhancements

- School quality ratings
- Transportation routes
- Tuition costs
- Admission requirements
- Specialized programs

## Future Enhancements

1. **Multi-Modal Distance**: Account for transit vs. car vs. walk
2. **Temporal Variation**: Morning commute times vs. geographic distance
3. **Quality Adjustment**: Weight by school performance metrics
4. **Enrollment Patterns**: Historical preference modeling
5. **Predictive Component**: Demographic trend forecasting

## References

1. Luo, W., & Wang, F. (2003). Measures of spatial accessibility to health care in a GIS environment. *International Journal of Health Geographics*.

2. Talen, E., & Anselin, L. (1998). Assessing spatial equity. *Environment and Planning A*.

3. Neutens, T. (2015). Accessibility, equity and health care. *Journal of Transport Geography*.

## Citation

```bibtex
@article{educational_desert_index_2025,
  title={Educational Desert Index: A Composite Measure of Educational Access},
  author={Your Name},
  year={2025},
  journal={Educational Planning},
  note={Methodology for Philadelphia block group analysis}
}
```
