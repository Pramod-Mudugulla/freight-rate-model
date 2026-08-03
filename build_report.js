const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle,
} = require("docx");

const PAGE_W = 12240, PAGE_H = 15840; // US Letter, DXA
const CONTENT_W = PAGE_W - 2 * 1440; // 1" margins

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 140 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function img(path, widthPx, capt) {
  const dims = imgDims[path];
  const w = Math.min(CONTENT_W, 620 * (widthPx || 1));
  const h = w * (dims.h / dims.w);
  const children = [
    new Paragraph({
      children: [new ImageRun({ data: fs.readFileSync(path), transformation: { width: w / 15, height: h / 15 }, type: "png" })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 100, after: 60 },
    }),
  ];
  if (capt) {
    children.push(new Paragraph({
      children: [new TextRun({ text: capt, italics: true, size: 18, color: "555555" })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
    }));
  }
  return children;
}

const imgDims = {
  "outputs/scorer_results/candidate_december.png": { w: 1922, h: 626 },
  "report_assets/01_distance_vs_rate.png": { w: 1032, h: 660 },
  "report_assets/02_rpm_distribution.png": { w: 1035, h: 660 },
  "report_assets/03_market_index_seasonality.png": { w: 1099, h: 660 },
  "report_assets/04_feature_importance.png": { w: 1035, h: 660 },
};

function metricsTable() {
  const header = ["Model", "MAE", "RMSE", "MAPE", "R²"].map(t =>
    new TableCell({
      width: { size: CONTENT_W / 5, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "064A56" },
      children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, color: "FFFFFF" })] })],
    })
  );
  const rows = [
    ["Baseline (median $/mile × distance)", "$256.95", "$684.25", "11.63%", "0.799"],
    ["Linear regression", "$187.08", "$652.58", "10.10%", "0.817"],
    ["GBM full  (used for validation.csv)", "$111.62", "$635.85", "4.78%", "0.826"],
    ["GBM lite  (used for December chart)", "$109.64", "$634.59", "4.80%", "0.827"],
  ];
  const bodyRows = rows.map((r, i) =>
    new TableRow({
      children: r.map((c, j) =>
        new TableCell({
          width: { size: CONTENT_W / 5, type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill: i >= 2 ? "E7F0F1" : "FFFFFF" },
          children: [new Paragraph({ children: [new TextRun({ text: c, bold: j === 0 && i >= 2 })] })],
        })
      ),
    })
  );
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W / 5, CONTENT_W / 5, CONTENT_W / 5, CONTENT_W / 5, CONTENT_W / 5],
    rows: [new TableRow({ children: header }), ...bodyRows],
  });
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children: [
      new Paragraph({ children: [new TextRun({ text: "Freight Rate Prediction — Assessment Report", bold: true, size: 40, color: "064A56" })], spacing: { after: 80 } }),
      new Paragraph({ children: [new TextRun({ text: "Machine Learning Engineer take-home", italics: true, size: 22, color: "555555" })], spacing: { after: 300 } }),

      h1("1. Objective"),
      p("Predict the posted rate ($) for a freight load from its pickup/delivery location, distance, equipment type, weight, date, and two market signals (market_index, quote_signal). Deliverables: predictions for 12,000 validation loads, and a 31-day predicted-rate chart for a single fixed lane in December 2025."),

      h1("2. Exploring the data & data-quality issues found"),
      p("train_test.csv has 48,000 labeled loads spanning 2025-01-01 to 2025-10-31. Before modeling, I checked distributions, missing values, and relationships between each feature and the target."),
      ...img("report_assets/01_distance_vs_rate.png", 0.85, "Distance explains most of the variance in rate on its own (~0.91 correlation)."),
      p("Issues identified and how I addressed each:", { bold: true }),
      bullet("weight: ~0.6% of rows missing, another ~0.6% negative (plausible sign-flip data-entry error — magnitudes look otherwise normal). Fix: take the absolute value, then fill missing with the median."),
      bullet("market_index: ~0.8% missing. Fix: fill with the median."),
      bullet("Rate-per-mile has a long right tail — about 1.4% of loads sit far outside the typical $1.50–$3.00/mile band, most likely genuine spot-rate spikes rather than errors, so I didn't delete them. Instead I trained the model with an absolute-error (MAE) loss instead of the default squared-error loss, which is far less sensitive to a handful of extreme values."),
      ...img("report_assets/02_rpm_distribution.png", 0.85, "Rate-per-mile distribution with the 1st/99th-percentile outlier band marked."),
      bullet("december_chart_inputs.csv has a different schema than validation.csv — it has no market_index, no quote_signal, and no lat/lon columns at all, only city names. I built a city → (lat, lon) lookup table from train_test.csv (which has one fixed coordinate per city) to backfill coordinates, and trained a second, reduced-feature model that only relies on columns present in every file (see Section 4)."),
      bullet("validation.csv contains 8 cities that never appear in train_test.csv (e.g. Chicago, Charlotte, Norfolk). This ruled out using the raw city name as a categorical feature — the model would have no learned behavior for an unseen city. I used latitude/longitude instead, which generalizes automatically to new locations."),
      p("I also checked market_index over time and found a clear, smooth seasonal cycle — it rises through spring, peaks early summer, dips in early autumn, and climbs again into November. This is useful context, though (as shown in Section 5) the trained model ended up leaning on it only lightly compared to distance."),
      ...img("report_assets/03_market_index_seasonality.png", 0.85, "Weekly average market_index across the training period — a clear annual cycle, not noise."),

      h1("3. Train / validation split approach"),
      p("train_test.csv only covers Jan–Oct 2025, but the actual task is forecasting Nov–Dec 2025 — dates the model has never seen. A random 80/20 shuffle would let the model learn from data surrounding a held-out day, overstating how well it would really do on the future."),
      p("Instead I split by date: the earliest ~80% of days train the model, and the most recent ~20% (Sept 1 onward) are held out purely for evaluation. That mirrors the real forecasting task. After confirming the model clears the baselines on this holdout, I retrain on 100% of train_test.csv (train + holdout combined) before generating the final submission predictions, so no labeled data goes unused."),

      h1("4. Model choice"),
      p("I used scikit-learn's HistGradientBoostingRegressor — a gradient-boosted decision tree model, the same family as XGBoost/LightGBM, built into scikit-learn so it adds no extra dependencies."),
      p("Why trees over a simple linear model: several features have non-linear relationships with rate. For example, quote_signal doesn't move the rate steadily — both low and high values push it up, while mid-range values push it down (a U-shape). A tree-based model captures interactions and non-linear effects like this automatically, without needing me to hand-engineer interaction terms."),
      p("Why two models instead of one: since validation.csv includes market_index/quote_signal but december_chart_inputs.csv does not, I trained:"),
      bullet("model_full — every available feature (incl. market_index, quote_signal) → used for the 12,000 validation predictions"),
      bullet("model_lite — only features present in every file → used for the December chart"),
      p("Permutation importance on the holdout set confirms this split costs very little accuracy: market_index and quote_signal barely move the needle once distance, equipment, and weight are already in the model."),
      ...img("report_assets/04_feature_importance.png", 0.85, "Permutation importance (GBM full): distance dominates; market signals contribute marginally."),

      h1("5. Results"),
      p("Metrics on the time-based holdout (Sept 1 – Oct 31, 2025), compared against two baselines:"),
      metricsTable(),
      new Paragraph({ text: "", spacing: { after: 200 } }),
      p("The gradient-boosted model roughly halves the error of a naive $/mile baseline and gets within ~5% of the true rate on average (MAPE). Model_full and model_lite perform almost identically, confirming that dropping market_index/quote_signal for the December predictions is a safe trade-off."),

      h1("6. December 2025 predicted-rate chart"),
      p("Fixed lane: Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb — only the date changes across the 31 rows. Generated by the provided score.py from december_chart_inputs.csv after filling in predicted_rate with model_lite."),
      ...img("outputs/scorer_results/candidate_december.png", 1.0, "Candidate: December 2025 predicted load rate (score.py output)."),
      p("The model didn't learn a strong month-to-month trend from day_of_year (its permutation importance came out at ~0) — most of the variation across the year in this dataset is really carried by market_index and quote_signal, and average rate-per-mile only moved modestly across months. What the model did pick up is a mild day-of-week pattern, which produces the small weekly wobble seen above rather than a flat line."),

      h1("7. Repository & how to reproduce"),
      p("Code, README with run instructions, and requirements.txt are included in the submitted GitHub repository. In short:"),
      bullet("pip install -r requirements.txt"),
      bullet("python src/train.py     (trains + evaluates both models, saves them)"),
      bullet("python src/predict.py   (writes validation_predictions.csv and the filled December file)"),
      bullet("python score.py --predictions ... --december-predictions ...   (official validation + chart)"),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("outputs/Freight_Rate_Report.docx", buf);
  console.log("wrote outputs/Freight_Rate_Report.docx");
});
