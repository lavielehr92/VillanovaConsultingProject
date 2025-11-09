"""
educational_desert_index.py

Detects educational deserts inside a city by combining:
1) Two-Step Floating Catchment Accessibility (2SFCA) supply-to-demand,
2) Gravity-based access (distance-decay),
3) Nearest-school distance,
4) Neighborhood need (e.g., poverty rate, % adults without HS).

Outputs a 0–100 Educational Desert Index (EDI) per block group.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Distance helpers
# -----------------------------
def _to_rad(deg: np.ndarray) -> np.ndarray:
    return np.deg2rad(deg.astype(float))

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Vectorized haversine distance (km).
    """
    R = 6371.0088
    lat1 = _to_rad(np.asarray(lat1))
    lon1 = _to_rad(np.asarray(lon1))
    lat2 = _to_rad(np.asarray(lat2))
    lon2 = _to_rad(np.asarray(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def pairwise_dist_km(pointsA: np.ndarray, pointsB: np.ndarray, chunk: int = 5000) -> np.ndarray:
    """
    Computes pairwise distances between A (n x 2) and B (m x 2) in km.
    Chunked to reduce peak memory.
    """
    n = pointsA.shape[0]
    m = pointsB.shape[0]
    out = np.empty((n, m), dtype=float)
    for i in range(0, n, chunk):
        i2 = min(i + chunk, n)
        latA = pointsA[i:i2, 0][:, None]
        lonA = pointsA[i:i2, 1][:, None]
        latB = pointsB[:, 0][None, :]
        lonB = pointsB[:, 1][None, :]
        out[i:i2, :] = haversine_km(latA, lonA, latB, lonB)
    return out

# -----------------------------
# Core access metrics
# -----------------------------
def nearest_school_distance_km(bg_df: pd.DataFrame, schools_df: pd.DataFrame) -> pd.Series:
    A = bg_df[["lat", "lon"]].to_numpy()
    B = schools_df[["lat", "lon"]].to_numpy()
    D = pairwise_dist_km(A, B)
    return pd.Series(D.min(axis=1), index=bg_df.index, name="nearest_km")

def gravity_access(bg_df: pd.DataFrame,
                   schools_df: pd.DataFrame,
                   decay_km: float = 2.0) -> pd.Series:
    """
    Gravity model with exponential distance decay.
    Higher = better access. We invert for desert scoring later.
    """
    A = bg_df[["lat", "lon"]].to_numpy()
    B = schools_df[["lat", "lon"]].to_numpy()
    cap = schools_df["capacity"].replace({0: np.nan}).fillna(0.0).to_numpy()
    D = pairwise_dist_km(A, B)
    # Avoid overflow if decay_km is tiny
    weights = np.exp(-D / max(decay_km, 1e-6))
    access = (weights * cap).sum(axis=1)
    return pd.Series(access, index=bg_df.index, name="gravity_access")

def two_step_fca(bg_df: pd.DataFrame,
                 schools_df: pd.DataFrame,
                 catchment_km: float = 2.0) -> pd.Series:
    """
    Two-Step Floating Catchment Accessibility (2SFCA).

    Step 1: For each school j, compute R_j = C_j / sum_i P_i (for i within catchment of j)
    Step 2: For each BG i, S_i = sum_j R_j (for j within catchment of i)

    Returns S_i where higher = better access (more seats per nearby student).
    """
    A = bg_df[["lat", "lon"]].to_numpy()
    B = schools_df[["lat", "lon"]].to_numpy()
    D = pairwise_dist_km(B, A)  # schools x bg
    within = (D <= catchment_km).astype(float)

    P_i = bg_df["k12_pop"].clip(lower=0.0).to_numpy()
    C_j = schools_df["capacity"].clip(lower=0.0).to_numpy()

    demand_in_catch = within @ P_i  # length m_schools
    # Avoid division by zero
    R_j = np.divide(C_j, np.where(demand_in_catch > 0, demand_in_catch, np.nan), where=~np.isnan(demand_in_catch))
    R_j = np.nan_to_num(R_j, nan=0.0)

    # Step 2: For each BG, sum R_j for schools within catchment
    D2 = pairwise_dist_km(A, B)  # bg x schools
    within2 = (D2 <= catchment_km).astype(float)
    S_i = within2 @ R_j  # length n_bg
    return pd.Series(S_i, index=bg_df.index, name="fca_supply_demand")

# -----------------------------
# Educational Desert Index (EDI)
# -----------------------------
def compute_edi(
    bg_df: pd.DataFrame,
    schools_df: pd.DataFrame,
    catchment_km: float = 2.0,
    gravity_decay_km: float = 2.0,
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Builds an Educational Desert Index = 0..100 where higher = more desert-like.

    Components (all scaled 0..1 before weighting):
      - nearest_km (higher is worse)
      - 1 - gravity_access_norm (lower access is worse)
      - 1 - fca_supply_demand_norm (lower supply/demand is worse)
      - poverty_rate (normalized if in 0..100)
      - pct_lt_hs (normalized if in 0..100)

    Default weights sum to 1. Adjust as needed.
    """
    df = bg_df.copy()

    # Normalize need metrics to 0..1 if they are in 0..100
    for col in ["poverty_rate", "pct_lt_hs"]:
        if col in df.columns:
            series = df[col].astype(float)
            if series.max() > 1.0:
                series = series / 100.0
            df[col] = series.clip(0.0, 1.0)
        else:
            df[col] = 0.0  # if not provided, neutral

    # Compute components
    df["nearest_km"] = nearest_school_distance_km(df, schools_df)
    df["gravity_access"] = gravity_access(df, schools_df, decay_km=gravity_decay_km)
    df["fca_supply_demand"] = two_step_fca(df, schools_df, catchment_km=catchment_km)

    # Scale positive-access metrics for inversion
    scaler_pos = MinMaxScaler(feature_range=(0, 1))
    for col in ["gravity_access", "fca_supply_demand"]:
        vals = df[[col]].to_numpy()
        if np.all(vals == vals[0]):  # constant column
            df[col + "_norm"] = 0.0
        else:
            df[col + "_norm"] = scaler_pos.fit_transform(vals)

    # Scale nearest distance as a burden (already a "worse when larger" measure)
    scaler_burden = MinMaxScaler(feature_range=(0, 1))
    vals = df[["nearest_km"]].to_numpy()
    if np.all(vals == vals[0]):
        df["nearest_norm"] = 0.0
    else:
        df["nearest_norm"] = scaler_burden.fit_transform(vals)

    # Default weights
    if weights is None:
        weights = {
            "nearest_norm": 0.20,
            "gravity_inv": 0.25,
            "fca_inv": 0.30,
            "poverty_rate": 0.15,
            "pct_lt_hs": 0.10,
        }

    # Build inversions where needed (low access => worse)
    df["gravity_inv"] = 1.0 - df["gravity_access_norm"]
    df["fca_inv"] = 1.0 - df["fca_supply_demand_norm"]

    # Weighted sum, then scale to 0..100
    components = {
        "nearest_norm": df["nearest_norm"],
        "gravity_inv": df["gravity_inv"],
        "fca_inv": df["fca_inv"],
        "poverty_rate": df["poverty_rate"],
        "pct_lt_hs": df["pct_lt_hs"],
    }

    # Ensure weights sum to 1
    wsum = sum(weights.get(k, 0.0) for k in components.keys())
    if wsum <= 0:
        raise ValueError("Weights must sum to a positive value.")
    norm_weights = {k: weights.get(k, 0.0) / wsum for k in components.keys()}

    edi_0_1 = sum(norm_weights[k] * components[k] for k in components.keys())
    edi_0_100 = (edi_0_1 * 100.0).rename("edi")

    # Helpful outputs
    out_cols = [
        "geoid_bg", "lat", "lon", "k12_pop",
        "poverty_rate", "pct_lt_hs",
        "nearest_km",
        "gravity_access", "gravity_access_norm",
        "fca_supply_demand", "fca_supply_demand_norm",
        "edi",
    ]
    df = df.assign(edi=edi_0_100)
    keep = [c for c in out_cols if c in df.columns]
    return df[keep].sort_values("edi", ascending=False)