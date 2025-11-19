import pandas as pd

def compute_legitimate_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean 'is_legit' column to the demographics DataFrame.

    Criteria for 'is_legit':
    - Has valid lat/lon coordinates or a valid geometry
    - total_pop > 0
    - k12_pop is not NaN and >= 0
    - income is not NaN and >= 0 (we don't exclude low incomes, only missing/invalid)
    - not flagged as suppressed or clearly invalid values (e.g., negative numbers)
    """
    df = df.copy()
    if 'lat' not in df.columns or 'lon' not in df.columns:
        if 'geometry' in df.columns:
            try:
                df['lat'] = df['geometry'].centroid.y
                df['lon'] = df['geometry'].centroid.x
            except Exception:
                df['lat'] = pd.NA
                df['lon'] = pd.NA
        else:
            df['lat'] = df.get('lat')
            df['lon'] = df.get('lon')

    df['is_legit'] = True
    df['is_legit'] = df['is_legit'] & (pd.to_numeric(df.get('total_pop', 0), errors='coerce') > 0)
    df['is_legit'] = df['is_legit'] & pd.to_numeric(df.get('k12_pop', 0), errors='coerce').notna()
    df['is_legit'] = df['is_legit'] & (pd.to_numeric(df.get('k12_pop', 0), errors='coerce') >= 0)
    df['is_legit'] = df['is_legit'] & pd.to_numeric(df.get('income', 0), errors='coerce').notna()
    df['is_legit'] = df['is_legit'] & (pd.to_numeric(df.get('income', -1), errors='coerce').fillna(-1) >= 0)
    df['is_legit'] = df['is_legit'] & df['lat'].notna() & df['lon'].notna()
    if '_suppressed' in df.columns:
        df['is_legit'] = df['is_legit'] & (~df['_suppressed'].astype(bool))
    return df
