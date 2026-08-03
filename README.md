# Freight Rate Prediction

Predicts `posted_rate` for freight loads (trucking spot-market rate) from load
features: pickup/delivery city, distance, equipment type, weight, date, and
two market signals (`market_index`, `quote_signal`).

## Quick start

```bash
python -m pip install -r requirements.txt

# 1. Train both models (reads data/train_test.csv, writes models/*.joblib)
python src/train.py

# 2. Generate both required prediction files (writes outputs/*.csv)
python src/predict.py

# 3. Run the official scorer (validates files + builds the December chart)
python -m pip install -r requirements.txt   # scorer needs pandas/numpy/matplotlib too
python score.py --predictions outputs/validation_predictions.csv \
                 --december-predictions outputs/december_chart_inputs.csv \
                 --output-dir outputs/scorer_results
```

Outputs land in `outputs/`:
- `validation_predictions.csv` — the 12,000-row submission file
- `december_chart_inputs.csv` — December file with `predicted_rate` filled in
- `scorer_results/candidate_december.png` — the chart

## Project layout

```
data/            input CSVs (as provided)
src/
  features.py    all data cleaning + feature engineering (shared by train & predict)
  train.py       trains + evaluates the models, saves them to models/
  predict.py     loads the saved models, writes the two submission files
models/          saved model files + metrics (created by train.py)
outputs/         final CSVs + chart (created by predict.py / score.py)
report_assets/   charts used in the report
```

## The approach

**The problem:** given a load's pickup/delivery, distance, weight, equipment
type, and date, predict what it will cost (`posted_rate`).

**Why gradient-boosted trees:** I checked how strongly the rate relates to each
input. Distance is overwhelmingly the biggest driver (~0.91 correlation on its
own). But a few other signals matter in a *non-linear* way — for example,
`quote_signal` doesn't move the rate up steadily; low AND high values both
push rate up a bit, medium values push it down (a "U-shape"). A plain linear
regression can't represent that; a tree-based model can, automatically,
without me hand-crafting interaction terms. I used scikit-learn's
`HistGradientBoostingRegressor` — same family of algorithm as XGBoost/LightGBM,
but it ships inside scikit-learn so there's nothing extra to install.

**Why two models instead of one:** `validation.csv` includes `market_index`
and `quote_signal`. `december_chart_inputs.csv` does **not** — it only has
6 columns and neither of those two. Rather than force one model to guess with
fewer inputs than it was trained on, I trained:
- `model_full` — every available feature, used for the validation predictions
- `model_lite` — only the features that exist in *every* file, used for December

Feature importance showed `market_index`/`quote_signal` barely move the
needle anyway (both models land within 0.1 points of MAPE of each other), so
this split doesn't cost meaningful accuracy — see `models/feature_importance.csv`.

**How the data was split for evaluation:** `train_test.csv` only covers
Jan 1 – Oct 31, 2025, but the real task is forecasting Nov/Dec 2025 — dates the
model has never seen. So instead of a random 80/20 shuffle (which would let
the model "see" data from both before and after a held-out day), I split by
**date**: train on the first ~80% of days, evaluate on the most recent ~20%
(Sept 1 onward). That's a fair proxy for "how well will this do on the future,"
which is exactly what December is. After confirming the model beats simple
baselines on that holdout, I retrain on 100% of `train_test.csv` before
generating the final predictions, so nothing is wasted.

**Data-quality issues found and how I handled them:**
1. `weight` — ~0.6% of rows are missing, and another ~0.6% are *negative*
   (a plausible sign-flip data-entry bug, since the magnitudes look normal).
   Fix: take the absolute value, then fill missing with the median.
2. `market_index` — ~0.8% missing. Fix: fill with the median.
3. Rate-per-mile has a long tail (~1.4% of rows sit far outside the typical
   $1.50–$3/mile range — likely genuine rate spikes or noisy entries).
   Rather than deleting that data (which could be real spot-market spikes),
   I trained with an **absolute-error loss** (MAE) instead of the default
   squared-error loss, which is much less sensitive to a handful of extreme
   values.
4. `december_chart_inputs.csv` has **no lat/lon columns**, only city names —
   different from every other file. Since `train_test.csv` has one fixed
   lat/lon per city, I built a city→coordinates lookup from it and used that
   to fill in December's lat/lon rather than dropping the feature.
5. `validation.csv` contains 8 cities that never appear in `train_test.csv`.
   That ruled out using city name as a raw category (the model would have no
   idea what to do with an unseen city). I used latitude/longitude instead,
   which generalizes to new cities automatically.

**Results on the time-based holdout** (`models/holdout_metrics.csv`):

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Baseline: median $/mile × distance | $256.95 | $684.25 | 11.63% | 0.799 |
| Linear regression | $187.08 | $652.58 | 10.10% | 0.817 |
| **GBM full (used for validation.csv)** | **$111.62** | $635.85 | **4.78%** | 0.826 |
| **GBM lite (used for December)** | **$109.64** | $634.59 | **4.80%** | 0.827 |

The gradient-boosted model roughly halves the error of a naive baseline and
gets the typical prediction within ~5% of the true rate.

## What I'd say if asked "what would you improve with more time?"
- Try LightGBM/XGBoost directly and tune more aggressively (I capped this to
  keep the dependency list to just scikit-learn/pandas/numpy/matplotlib).
- Model the outlier loads separately (e.g., a classifier that flags "likely
  spot-rate spike" and routes it to a wider prediction interval) instead of
  just downweighting them via the loss function.
- Add a lane-level historical average (not raw city name, but e.g. average
  $/mile seen historically for pickup-region → delivery-region) as an extra
  feature, with a fallback for unseen lanes.
