"""
Module C -- demand forecasting.

Two rules govern this module.

1. ALWAYS BEAT A BASELINE.
   A raw accuracy number is meaningless on its own. The ladder runs
   naive -> seasonal naive -> moving average -> Holt-Winters -> LightGBM, and
   the reported result is the improvement over SEASONAL NAIVE. If a gradient
   boosting model cannot beat "same day last week", it should not ship.

2. NEVER RANDOM-SPLIT A TIME SERIES.
   Validation is rolling-origin (expanding window): train on everything up to
   time t, predict the next 28 days, roll forward, repeat. A random KFold would
   let the model train on Friday to predict the preceding Thursday, which leaks
   the future and produces an accuracy number that collapses in production.

METRIC: WMAPE, not MAPE.
   WMAPE = sum|y - yhat| / sum|y|
   MAPE divides by each actual individually, so any zero-demand day is a divide
   by zero and any near-zero day produces an enormous percentage that swamps
   the average. Intermittent FMCG SKUs are full of zero days, so MAPE is
   unusable here. WMAPE pools the errors and divides once, which also weights
   high-volume SKUs appropriately.
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (FIGURES, FORECAST_HORIZON, N_CV_FOLDS, SEASONAL_PERIOD,
                    TABLES)

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------- metrics
def wmape(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom > 0 else np.nan


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def bias(y, yhat):
    """Mean signed error. Positive = under-forecasting (drives stockouts)."""
    return float(np.mean(y - yhat))


# --------------------------------------------------------------- CV splits
def rolling_origin_splits(dates: pd.DatetimeIndex, horizon: int, n_folds: int):
    """
    Yield (train_end_date, test_start, test_end) tuples, latest fold last.

    Fold k trains on [start, cutoff_k] and tests on the next `horizon` days.
    The training window EXPANDS rather than slides, so each fold uses all
    history available at that point in time -- the same information a model
    retrained in production would have.
    """
    last = dates.max()
    for k in range(n_folds, 0, -1):
        test_end = last - pd.Timedelta(days=(k - 1) * horizon)
        test_start = test_end - pd.Timedelta(days=horizon - 1)
        train_end = test_start - pd.Timedelta(days=1)
        yield train_end, test_start, test_end


# --------------------------------------------------------- feature building
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag and rolling features for the global LightGBM model.

    Every feature is shifted by at least 1 day. A rolling mean computed on the
    current row would include today's demand, which is the value being
    predicted -- the classic leakage bug in time-series feature engineering.
    """
    d = df.sort_values(["product_id", "order_date"]).copy()
    g = d.groupby("product_id")["units"]

    for lag in (1, 7, 14, 28):
        d[f"lag_{lag}"] = g.shift(lag)

    for win in (7, 28):
        # shift(1) first, THEN roll: the window ends yesterday, not today.
        d[f"roll_mean_{win}"] = g.shift(1).rolling(win).mean().reset_index(drop=True)
        d[f"roll_std_{win}"] = g.shift(1).rolling(win).std().reset_index(drop=True)

    d["dow"] = d["order_date"].dt.dayofweek
    d["month"] = d["order_date"].dt.month
    d["week_of_year"] = d["order_date"].dt.isocalendar().week.astype(int)
    d["day_of_month"] = d["order_date"].dt.day
    return d


FEATURES = ["lag_1", "lag_7", "lag_14", "lag_28",
            "roll_mean_7", "roll_std_7", "roll_mean_28", "roll_std_28",
            "dow", "month", "week_of_year", "day_of_month", "sku_code"]


# ------------------------------------------------------------------ models
def fit_predict_lightgbm(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """
    ONE global model across all SKUs, with product_id as a categorical.

    A global model beats 120 per-SKU models here because sparse SKUs borrow
    strength from the shared weekly and seasonal patterns instead of trying to
    learn them from a handful of non-zero observations.
    """
    import lightgbm as lgb

    tr = train.dropna(subset=FEATURES)
    if len(tr) < 200:
        return np.full(len(test), train["units"].mean())

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9, random_state=42, verbose=-1,
    )
    model.fit(tr[FEATURES], tr["units"],
              categorical_feature=["sku_code", "dow", "month"])
    pred = model.predict(test[FEATURES].fillna(0))
    return np.clip(pred, 0, None)  # demand cannot be negative


def fit_predict_holt_winters(train_series: pd.Series, n_ahead: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # Reset to a clean positional index. Passing the parent frame's original
    # index makes statsmodels warn that it cannot infer a time index; the
    # values are already in date order so position carries the same meaning.
    s = pd.Series(train_series.to_numpy(dtype=float))
    # Needs at least two full seasonal cycles to estimate seasonal factors.
    if len(s) < 2 * SEASONAL_PERIOD + 2 or s.sum() == 0:
        return np.full(n_ahead, s.mean() if len(s) else 0.0)
    try:
        fit = ExponentialSmoothing(
            s, trend="add", seasonal="add",
            seasonal_periods=SEASONAL_PERIOD,
            initialization_method="estimated",
        ).fit(optimized=True)
        return np.clip(fit.forecast(n_ahead).to_numpy(), 0, None)
    except Exception:
        return np.full(n_ahead, s.mean())


# --------------------------------------------------------------- evaluation
def run(daily: pd.DataFrame, sku_subset: list[str] | None = None) -> dict:
    d = daily.copy()
    if sku_subset is not None:
        d = d.loc[d["product_id"].isin(sku_subset)]

    d["sku_code"] = d["product_id"].astype("category").cat.codes
    feat = make_features(d)
    dates = pd.DatetimeIndex(feat["order_date"].unique())

    rows, preds_store = [], []

    for fold, (train_end, test_start, test_end) in enumerate(
        rolling_origin_splits(dates, FORECAST_HORIZON, N_CV_FOLDS), start=1
    ):
        train = feat.loc[feat["order_date"] <= train_end]
        test = feat.loc[(feat["order_date"] >= test_start)
                        & (feat["order_date"] <= test_end)].copy()
        if test.empty or train.empty:
            continue

        y = test["units"].to_numpy(dtype=float)
        fold_preds = {}

        # Baseline 1: naive -- yesterday's value carried forward.
        fold_preds["Naive"] = test["lag_1"].fillna(0).to_numpy(dtype=float)

        # Baseline 2: seasonal naive -- same weekday last week. The bar to beat.
        fold_preds["SeasonalNaive"] = test["lag_7"].fillna(0).to_numpy(dtype=float)

        # Baseline 3: 28-day moving average.
        fold_preds["MovingAvg28"] = test["roll_mean_28"].fillna(0).to_numpy(dtype=float)

        # Holt-Winters, fitted per SKU on that SKU's own history.
        hw = np.zeros(len(test))
        for sku, grp in test.groupby("product_id", sort=False):
            hist = (train.loc[train["product_id"] == sku]
                    .sort_values("order_date")["units"])
            fc = fit_predict_holt_winters(hist, len(grp))
            hw[test["product_id"].to_numpy() == sku] = fc
        fold_preds["HoltWinters"] = hw

        # Global gradient boosting.
        fold_preds["LightGBM"] = fit_predict_lightgbm(train, test)

        for name, yhat in fold_preds.items():
            rows.append({
                "fold": fold, "model": name,
                "test_start": test_start.date(), "test_end": test_end.date(),
                "wmape": wmape(y, yhat), "mae": mae(y, yhat),
                "rmse": rmse(y, yhat), "bias": bias(y, yhat),
                "n_obs": len(y),
            })

        keep = test[["product_id", "order_date", "units"]].copy()
        keep["fold"] = fold
        for name, yhat in fold_preds.items():
            keep[name] = yhat
        preds_store.append(keep)

    results = pd.DataFrame(rows)
    preds = pd.concat(preds_store, ignore_index=True) if preds_store else pd.DataFrame()

    summary = (
        results.groupby("model")
        .agg(wmape=("wmape", "mean"), mae=("mae", "mean"),
             rmse=("rmse", "mean"), bias=("bias", "mean"))
        .sort_values("wmape")
        .reset_index()
    )
    base = summary.loc[summary["model"] == "SeasonalNaive", "wmape"].iloc[0]
    summary["vs_seasonal_naive_pct"] = (base - summary["wmape"]) / base * 100

    results.to_csv(TABLES / "forecast_fold_results.csv", index=False)
    summary.to_csv(TABLES / "forecast_model_summary.csv", index=False)
    if not preds.empty:
        preds.to_csv(TABLES / "forecast_predictions.csv", index=False)

    _plot(summary, preds)

    best = summary.iloc[0]
    return {"results": results, "summary": summary, "predictions": preds,
            "headline": {
                "best_model": str(best["model"]),
                "best_wmape": float(best["wmape"]),
                "seasonal_naive_wmape": float(base),
                "improvement_pct": float(best["vs_seasonal_naive_pct"]),
                "n_skus": int(d["product_id"].nunique()),
                "n_folds": int(results["fold"].nunique()),
            }}


def _plot(summary: pd.DataFrame, preds: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))

    a = ax[0]
    d = summary.sort_values("wmape", ascending=False)
    cols = ["#55a868" if m == summary.iloc[0]["model"] else "#4c72b0"
            for m in d["model"]]
    a.barh(d["model"], d["wmape"], color=cols)
    for i, v in enumerate(d["wmape"]):
        a.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    a.set_xlabel("WMAPE (lower is better)")
    a.set_title("Model comparison, mean over rolling-origin folds")

    a = ax[1]
    if not preds.empty:
        agg = (preds.groupby("order_date")
               .agg(actual=("units", "sum"),
                    forecast=(summary.iloc[0]["model"], "sum"))
               .reset_index())
        a.plot(agg["order_date"], agg["actual"], label="actual",
               color="#222", lw=1.6)
        a.plot(agg["order_date"], agg["forecast"],
               label=f"forecast ({summary.iloc[0]['model']})",
               color="#c44e52", lw=1.6, ls="--")
        a.legend(fontsize=8)
        a.tick_params(axis="x", rotation=45, labelsize=8)
    a.set_title("Total daily demand: actual vs forecast (held-out folds)")
    a.set_ylabel("units")

    fig.tight_layout()
    fig.savefig(FIGURES / "03_forecast_performance.png")
    plt.close(fig)


if __name__ == "__main__":
    daily = pd.read_csv("../data/processed/daily_demand.csv",
                        parse_dates=["order_date"])
    seg = pd.read_csv("../outputs/tables/abc_xyz_sku_level.csv")
    focus = seg.loc[seg["abc"].isin(["A", "B"]), "product_id"].astype(str).tolist()
    res = run(daily, sku_subset=focus)
    print(res["summary"].to_string(index=False))
    h = res["headline"]
    print(f"\nBest: {h['best_model']}  WMAPE={h['best_wmape']:.4f}  "
          f"({h['improvement_pct']:.1f}% better than seasonal naive)")
