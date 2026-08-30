# FMCG Supply Chain & Demand Analytics

End-to-end supply chain analytics on ~237K order lines: delivery performance
diagnostics, ABC-XYZ SKU segmentation, demand forecasting with rolling-origin
validation, and safety-stock optimisation validated by inventory simulation.

The output is a stocking policy, not a dashboard. Each module answers a
question a planner actually asks: *where are we failing on service, which SKUs
deserve attention, what will demand be, and how much stock should we hold.*

---

## Headline results

| Question | Finding |
|---|---|
| How bad is delivery? | **57.8%** of fulfilled lines arrive late, averaging **1.79 days** past the promised date |
| Where does it concentrate? | **Standard Class** has a 64.1% late rate and carries **66.5% of all late orders** |
| Is it a product problem? | **No.** Late delivery is independent of product category (chi-square p = 0.32); it is driven by shipping mode (Cramer's V = 0.246) and region (0.186) |
| Which SKUs matter? | **25 of 120 SKUs (21%)** generate **79% of revenue**; 39 SKUs are erratic (weekly CV >= 1.0) |
| Can we forecast demand? | Holt-Winters reaches **WMAPE 0.703**, **24.6% better** than a seasonal-naive baseline |
| How much stock should we hold? | Segment-differentiated safety stock cuts inventory capital **34.6%** and total cost **24.4%** vs a flat 21-day buffer, holding fill rate at **99.17%** |

The last row is the commercial point. Against a *uniform* 99% service rule the
segmented policy still saves **7.2%** of capital, which shows the gain comes
from differentiating service levels by segment rather than from simply holding
less stock everywhere.

---

## Data provenance — read this first

This repo runs on either of two inputs:

1. **Real data (recommended).** Download the
   [DataCo Smart Supply Chain dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
   from Kaggle and place `DataCoSupplyChainDataset.csv` in `data/raw/`.
   The pipeline detects it, maps its column names, and uses it automatically.

2. **Synthetic fallback (what generated the numbers above).** If no Kaggle file
   is present, `src/make_synthetic_data.py` builds a schema-compatible dataset
   so the repo runs on a fresh clone with no download.

**The figures in this README came from the synthetic generator.** They are real
outputs of real code, but they are not observations about a real supply chain.
Drop in the Kaggle file and re-run `python src/run_all.py` to regenerate every
number against real data.

---

## Method

### 1. Delivery performance (`src/otif_analysis.py`)

OTIF treats a line as a success only if it was fulfilled *and* arrived by the
promised date. Cancelled and suspected-fraud lines never shipped, so they are
excluded from the denominator rather than counted as failures.

The module reports each segment's **share of all late orders**, not just its
late rate. A segment with a 90% failure rate on 200 lines is noise; Standard
Class at 64% on 106K lines is the problem. Chi-square tests confirm the
differences are not sampling noise, and Cramer's V is reported alongside
because with n > 200K almost any difference reaches significance.

### 2. ABC-XYZ segmentation (`src/abc_xyz.py`)

- **ABC** — sort SKUs by revenue, cut on *cumulative* share: A <= 80%, B <= 95%, C the rest.
- **XYZ** — coefficient of variation (CV = sigma/mu) of **weekly** demand: X < 0.5, Y < 1.0, Z >= 1.0.

Weekly rather than daily buckets, deliberately. Daily FMCG demand is full of
structural zeros, so a daily CV mostly measures order frequency instead of
genuine demand instability.

The 9-cell matrix maps to a policy: AX gets automated replenishment with a thin
buffer, AZ gets manual planner review, CZ is make-to-order or delist.

### 3. Demand forecasting (`src/forecasting.py`)

Baseline ladder: naive → seasonal naive → 28-day moving average → Holt-Winters
→ LightGBM (one global model, SKU as a categorical feature).

Two non-negotiables:

- **Validation is rolling-origin**, 3 folds × 28-day horizon, expanding
  training window. A random split would let the model train on Friday to
  predict the preceding Thursday.
- **The metric is WMAPE**, not MAPE. MAPE divides by each actual individually,
  so zero-demand days are divide-by-zero and near-zero days produce enormous
  percentages. WMAPE pools errors and divides once.

Every result is reported as improvement over **seasonal naive**, because a
model that cannot beat "same day last week" should not ship.

| Model | WMAPE | MAE | RMSE | Bias | vs seasonal naive |
|---|---:|---:|---:|---:|---:|
| Holt-Winters | 0.703 | 7.99 | 20.34 | +0.96 | **+24.6%** |
| LightGBM | 0.823 | 9.38 | 25.41 | +5.58 | +11.7% |
| Seasonal naive | 0.932 | 10.61 | 28.38 | +0.29 | baseline |
| Naive | 0.965 | 11.02 | 29.41 | +0.13 | −3.5% |
| Moving average (28d) | 1.321 | 15.02 | 32.71 | +5.47 | −41.7% |

Holt-Winters beating LightGBM is reported as found, not tuned away. At
SKU-day granularity the series are short and noisy, and per-SKU exponential
smoothing captures level plus weekly seasonality without needing to learn it
from features. LightGBM also shows a **positive bias of +5.58 units**,
i.e. systematic under-forecasting, which in an inventory context directly
causes stockouts — a strong reason to reject it here beyond the WMAPE gap.

### 4. Inventory optimisation (`src/inventory.py`)

Safety stock under joint demand and lead-time uncertainty:

```
SS  = z * sqrt( L_bar * sigma_d^2  +  d_bar^2 * sigma_L^2 )
ROP = d_bar * L_bar + SS
EOQ = sqrt( 2 * D * S / H )
```

The two terms under the root are independent risk sources and add as
*variances*, which is why the square root sits outside the sum. The common
shortcut `SS = z * sigma_d * sqrt(L)` drops lead-time variability and
systematically under-buffers.

Three policies are then simulated against held-out actual demand under a
continuous-review (s, Q) rule with stochastic lead times:

| Policy | Fill rate | Avg inventory value | Total cost |
|---|---:|---:|---:|
| Flat 21-day buffer | 99.92% | 137,565 | 49,016 |
| Uniform 99% | 99.70% | 96,888 | 38,872 |
| **Segment-optimised** | **99.17%** | **89,917** | **37,079** |

Simulation rather than formula because the formula gives *cycle service level*
(probability of no stockout in a cycle) while the business cares about *fill
rate* (fraction of units actually served). These differ, and the gap widens for
lumpy demand.

---

## Repo structure

```
fmcg-supply-chain-analytics/
├── src/
│   ├── config.py               # paths, cutoffs, business parameters
│   ├── make_synthetic_data.py  # DataCo-schema fallback generator
│   ├── data_prep.py            # load, clean, OTIF flags, daily demand grid
│   ├── otif_analysis.py        # Module A: delivery performance
│   ├── abc_xyz.py              # Module B: segmentation
│   ├── forecasting.py          # Module C: baseline ladder + rolling-origin CV
│   ├── inventory.py            # Module D: safety stock, EOQ, simulation
│   └── run_all.py              # orchestrator
├── sql/analysis_queries.sql    # CTEs + window functions, PostgreSQL
├── notebooks/                  # narrative walkthrough
├── outputs/
│   ├── figures/                # 4 diagnostic charts
│   ├── tables/                 # 15 result CSVs (Power BI sources)
│   ├── RESULTS.md              # auto-generated report
│   └── results.json
├── powerbi/DASHBOARD_GUIDE.md  # 3-page dashboard spec
└── INTERVIEW_PREP.md           # concepts, formulas, Q&A
```

## Running it

```bash
git clone <your-repo-url>
cd fmcg-supply-chain-analytics
pip install -r requirements.txt

# optional: place DataCoSupplyChainDataset.csv in data/raw/ for real data
python src/run_all.py
```

Runs in roughly two minutes on a laptop and regenerates every figure, table
and `outputs/RESULTS.md`.

## Stack

Python (pandas, NumPy, scikit-learn, statsmodels, LightGBM, SciPy, Matplotlib),
SQL (PostgreSQL window functions and CTEs), Power BI.

## Known limitations

- Lead times are parameterised (mean 7 days, sd 2) rather than observed; the
  source data records shipping duration but not supplier replenishment lead time.
- The simulation assumes lost sales, not backorders. Backordering would raise
  effective fill rate and lower the apparent cost of a thin buffer.
- Holding cost is modelled at 25% of unit cost annually, a standard planning
  assumption rather than a measured figure.
- Forecasting covers A and B class SKUs (59 of 120). C-class demand is too
  intermittent for the models used here; Croston's method would be the correct
  approach for that tail.
