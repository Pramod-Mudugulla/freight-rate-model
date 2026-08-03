"""
Train two models on data/train_test.csv:

  model_full.joblib  - uses ALL features (incl. market_index, quote_signal)
                        -> used later to predict validation.csv
  model_lite.joblib  - uses only the features common to every file
                        -> used later to predict december_chart_inputs.csv
                           (which doesn't have market_index / quote_signal)

Both are HistGradientBoostingRegressor (scikit-learn's gradient-boosted
trees implementation). Why this model:
  - Tabular data with non-linear effects and interactions (we saw quote_signal
    has a U-shaped relationship with rate-per-mile, not a straight line)
    -> tree-based models capture this without manual feature crosses.
  - Handles missing values natively, so no separate imputer is required
    inside the model (we still clean weight/market_index in features.py
    for transparency and consistency).
  - loss="absolute_error" (i.e. trains to minimize MAE, not squared error)
    so the ~1.4% of rows with extreme rate-per-mile outliers (likely spot-rate
    spikes / data noise) don't dominate training the way they would under a
    squared-error loss.

VALIDATION STRATEGY (important):
We do a TIME-BASED split, not a random shuffle. train_test.csv covers
2025-01-01 to 2025-10-31, and the real validation/December sets are ALL in
the future (Nov-Dec 2025). A random split would let the model "peek" at
patterns from dates surrounding a held-out row, which is not representative
of the actual forecasting task. So we hold out the most recent ~20% of dates
(2025-09-05 onward) as an internal test set, and train on everything before it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from features import build_full_features, build_lite_features, set_city_coords_reference  # noqa: E402

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def time_split(df: pd.DataFrame, holdout_frac: float = 0.2):
    df = df.sort_values("date")
    dates_sorted = np.sort(df["date"].unique())
    cutoff_date = dates_sorted[int(len(dates_sorted) * (1 - holdout_frac))]
    train = df[df["date"] < cutoff_date]
    test = df[df["date"] >= cutoff_date]
    return train, test, cutoff_date


def evaluate(name: str, y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:<28s} MAE=${mae:8.2f}   RMSE=${rmse:8.2f}   MAPE={mape:6.2f}%   R2={r2:.4f}")
    return {"model": name, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=8,
        max_iter=400,
        learning_rate=0.06,
        l2_regularization=0.1,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
        categorical_features="from_dtype",  # any pandas "category" dtype column is treated as categorical
        random_state=42,
    )


def main():
    raw = pd.read_csv(DATA / "train_test.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    set_city_coords_reference(raw)

    train_raw, test_raw, cutoff = time_split(raw, holdout_frac=0.2)
    cutoff_str = pd.Timestamp(cutoff).date()
    print(f"Time-based split: train={len(train_raw):,} rows (< {cutoff_str}), "
          f"holdout={len(test_raw):,} rows (>= {cutoff_str})\n")

    y_train, y_test = train_raw["posted_rate"].values, test_raw["posted_rate"].values

    results = []

    # ---- Naive baseline: median $/mile from training data * distance ----
    rate_per_mile = (train_raw["posted_rate"] / train_raw["distance"]).median()
    baseline_pred = test_raw["distance"].values * rate_per_mile
    results.append(evaluate("Baseline (median $/mile)", y_test, baseline_pred))

    # ---- Linear regression baseline (lite features) for comparison ----
    X_train_lite, X_test_lite = build_lite_features(train_raw), build_lite_features(test_raw)
    X_train_lite_ohe = pd.get_dummies(X_train_lite, columns=["equipment"])
    X_test_lite_ohe = pd.get_dummies(X_test_lite, columns=["equipment"]).reindex(columns=X_train_lite_ohe.columns, fill_value=0)
    lin = LinearRegression().fit(X_train_lite_ohe, y_train)
    results.append(evaluate("Linear regression (lite feats)", y_test, lin.predict(X_test_lite_ohe)))

    # ---- Main model: full features (used for validation.csv) ----
    X_train_full, X_test_full = build_full_features(train_raw), build_full_features(test_raw)
    for col in ["equipment"]:
        X_train_full[col] = X_train_full[col].astype("category")
        X_test_full[col] = pd.Categorical(X_test_full[col], categories=X_train_full[col].cat.categories)
    model_full = make_model()
    model_full.fit(X_train_full, y_train)
    results.append(evaluate("GBM full (-> validation.csv)", y_test, model_full.predict(X_test_full)))

    # ---- Lite model: reduced features (used for december_chart_inputs.csv) ----
    for col in ["equipment"]:
        X_train_lite[col] = X_train_lite[col].astype("category")
        X_test_lite[col] = pd.Categorical(X_test_lite[col], categories=X_train_lite[col].cat.categories)
    model_lite = make_model()
    model_lite.fit(X_train_lite, y_train)
    results.append(evaluate("GBM lite (-> december chart)", y_test, model_lite.predict(X_test_lite)))

    # ---- Permutation importance on the holdout set: which features actually matter ----
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(model_full, X_test_full, y_test, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error")
    importance_df = pd.DataFrame({
        "feature": X_test_full.columns,
        "importance_mae_increase": perm.importances_mean,
    }).sort_values("importance_mae_increase", ascending=False)
    print("\nPermutation importance (GBM full, on time-based holdout - how much worse MAE gets if a feature is shuffled):")
    print(importance_df.to_string(index=False))
    importance_df.to_csv(MODELS / "feature_importance.csv", index=False)

    # ---- Refit both models on ALL of train_test.csv (train + holdout) before shipping ----
    # Now that we've picked the model and confirmed it beats the baselines on the
    # time-based holdout, we retrain on 100% of the labeled data so the final
    # model has seen as much history as possible before predicting Nov/Dec.
    X_full_all = build_full_features(raw)
    X_full_all["equipment"] = X_full_all["equipment"].astype("category")
    final_full = make_model()
    final_full.fit(X_full_all, raw["posted_rate"].values)

    X_lite_all = build_lite_features(raw)
    X_lite_all["equipment"] = X_lite_all["equipment"].astype("category")
    final_lite = make_model()
    final_lite.fit(X_lite_all, raw["posted_rate"].values)

    joblib.dump(final_full, MODELS / "model_full.joblib")
    joblib.dump(final_lite, MODELS / "model_lite.joblib")
    joblib.dump(list(X_full_all.columns), MODELS / "model_full_columns.joblib")
    joblib.dump(list(X_lite_all.columns), MODELS / "model_lite_columns.joblib")

    pd.DataFrame(results).to_csv(MODELS / "holdout_metrics.csv", index=False)
    print(f"\nSaved models to {MODELS}/ and holdout metrics to {MODELS}/holdout_metrics.csv")


if __name__ == "__main__":
    main()
