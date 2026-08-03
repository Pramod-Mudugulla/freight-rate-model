"""
Load the two trained models and produce both deliverables:

  1. outputs/validation_predictions.csv   (12,000 rows, load_id + predicted_rate)
     -> uses model_full.joblib, since validation.csv HAS market_index/quote_signal

  2. outputs/december_chart_inputs.csv    (31 rows, original columns + predicted_rate filled)
     -> uses model_lite.joblib, since this file does NOT have market_index/quote_signal
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_full_features, build_lite_features, set_city_coords_reference  # noqa: E402

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def align_categorical(df: pd.DataFrame, ref_columns: list[str]) -> pd.DataFrame:
    df = df[ref_columns].copy()
    return df


def main():
    model_full = joblib.load(MODELS / "model_full.joblib")
    model_lite = joblib.load(MODELS / "model_lite.joblib")
    full_cols = joblib.load(MODELS / "model_full_columns.joblib")
    lite_cols = joblib.load(MODELS / "model_lite_columns.joblib")

    train_ref = pd.read_csv(DATA / "train_test.csv")
    set_city_coords_reference(train_ref)

    # --- 1) validation.csv -> validation_predictions.csv ---
    val = pd.read_csv(DATA / "validation.csv")
    X_val = build_full_features(val)
    X_val["equipment"] = X_val["equipment"].astype("category")
    X_val = align_categorical(X_val, full_cols)
    val_preds = model_full.predict(X_val)
    val_preds = val_preds.clip(min=1)  # score.py rejects non-positive rates

    out_val = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": val_preds.round(2)})
    out_val.to_csv(OUT / "validation_predictions.csv", index=False)
    print(f"Wrote {OUT / 'validation_predictions.csv'} ({len(out_val):,} rows)")

    # --- 2) december_chart_inputs.csv -> filled december_chart_inputs.csv ---
    dec = pd.read_csv(DATA / "december_chart_inputs.csv")
    X_dec = build_lite_features(dec)
    X_dec["equipment"] = X_dec["equipment"].astype("category")
    X_dec = align_categorical(X_dec, lite_cols)
    dec_preds = model_lite.predict(X_dec)
    dec_preds = dec_preds.clip(min=1)

    out_dec = dec.copy()
    out_dec["predicted_rate"] = dec_preds.round(2)
    out_dec.to_csv(OUT / "december_chart_inputs.csv", index=False)
    print(f"Wrote {OUT / 'december_chart_inputs.csv'} ({len(out_dec):,} rows)")
    print(out_dec[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
