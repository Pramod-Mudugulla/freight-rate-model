"""
Feature engineering shared by training and prediction.

Design note (important, explained in the report/Loom):
----------------------------------------------------------------
validation.csv has 13 input columns, including `market_index` and
`quote_signal`. december_chart_inputs.csv only has 6 input columns -
it does NOT include `market_index` or `quote_signal` at all.

So we can't use one feature set for both files. This module builds
two feature sets from the same cleaning logic:

  - build_full_features()  -> uses everything, incl. market_index / quote_signal
                               (used for the 12,000-row validation.csv predictions)
  - build_lite_features()  -> only uses columns that exist in BOTH files
                               (used for the December chart, since that file
                               never has market_index / quote_signal)

Both share the same cleaning rules so behavior stays consistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that exist in every input file we'll ever see (train, validation, december)
COMMON_NUMERIC = ["distance", "weight", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
CATEGORICAL = ["equipment"]

# Only present in train_test.csv and validation.csv
EXTRA_NUMERIC = ["market_index", "quote_signal"]


def _clean_weight(df: pd.DataFrame) -> pd.Series:
    """Weight has ~0.6% negative values (sign-flip data entry errors) and
    ~0.6% missing values. We take abs() and impute missing with the median."""
    weight = df["weight"].abs()
    weight = weight.fillna(weight.median())
    return weight


def _clean_market_index(df: pd.DataFrame) -> pd.Series:
    """~0.8% missing. Impute with median (index hovers in a fairly narrow band)."""
    return df["market_index"].fillna(df["market_index"].median())


def _date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the date into seasonality signals a tree model can use.
    Cyclical (sin/cos) encoding for month and day-of-week so e.g. Dec (12)
    and Jan (1) are recognized as adjacent instead of far apart.
    day_of_year is kept as a plain number so the model can pick up on the
    smooth yearly cycle we saw in market_index (rises to a spring/summer peak,
    dips in early autumn, climbs again into winter)."""
    dt = pd.to_datetime(df["date"])
    out = pd.DataFrame(index=df.index)
    out["day_of_year"] = dt.dt.dayofyear
    month = dt.dt.month
    dow = dt.dt.dayofweek
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return out


def _base_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = _clean_weight(df)
    df["equipment"] = df["equipment"].astype("category")
    return df


_CITY_COORDS = None


def set_city_coords_reference(train_df: pd.DataFrame) -> None:
    """Call once with train_test.csv to build a city -> (lat, lon) lookup.
    december_chart_inputs.csv only gives city names, not lat/lon columns,
    so we need this to derive lat/lon for that file. train_test.csv has a
    fixed one-to-one city -> coordinate mapping (verified during EDA), so
    this lookup is exact, not an approximation."""
    global _CITY_COORDS
    pickups = train_df[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
    )
    deliveries = train_df[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
    )
    _CITY_COORDS = pd.concat([pickups, deliveries]).drop_duplicates("city").set_index("city")


def _ensure_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """If lat/lon columns are missing (december_chart_inputs.csv case), look
    them up by city name from the reference table built off train_test.csv."""
    df = df.copy()
    needed = {"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"}
    if needed.issubset(df.columns):
        return df
    if _CITY_COORDS is None:
        raise RuntimeError(
            "lat/lon columns missing and no city reference set - call "
            "set_city_coords_reference(train_df) first."
        )
    df["pickup_lat"] = df["pickup"].map(_CITY_COORDS["lat"])
    df["pickup_lon"] = df["pickup"].map(_CITY_COORDS["lon"])
    df["delivery_lat"] = df["delivery"].map(_CITY_COORDS["lat"])
    df["delivery_lon"] = df["delivery"].map(_CITY_COORDS["lon"])
    return df


def build_lite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature set that only needs columns present in EVERY file
    (train_test, validation, and december_chart_inputs).
    Deliberately does NOT use pickup/delivery city names as categories:
    validation.csv contains 8 cities that never appear in train_test.csv,
    so a city-based feature would be undefined (missing) for ~12% of the
    validation rows. Lat/lon (numeric, always available or looked up by
    city name) generalize to new cities instead."""
    df = _ensure_latlon(df)
    df = _base_clean(df)
    feats = df[COMMON_NUMERIC + CATEGORICAL].copy()
    feats = pd.concat([feats, _date_features(df)], axis=1)
    return feats


def build_full_features(df: pd.DataFrame) -> pd.DataFrame:
    """Everything in build_lite_features(), plus market_index and quote_signal.
    Only usable on files that actually contain those two columns
    (train_test.csv, validation.csv) - NOT december_chart_inputs.csv."""
    df = _ensure_latlon(df)
    df = _base_clean(df)
    df["market_index"] = _clean_market_index(df)
    feats = df[COMMON_NUMERIC + CATEGORICAL + EXTRA_NUMERIC].copy()
    feats = pd.concat([feats, _date_features(df)], axis=1)
    return feats
