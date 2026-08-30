"""
Generate a synthetic order-line dataset that mirrors the DataCo Smart Supply
Chain schema (column names, shipping modes, scheduled lead times, delivery
statuses).

WHY THIS EXISTS
---------------
The real dataset is a Kaggle download and cannot be committed to the repo.
This generator lets the whole pipeline run end-to-end on a fresh clone. Every
downstream module is schema-driven, so dropping the real
`DataCoSupplyChainDataset.csv` into data/raw/ switches the analysis to real
data with no code change.

The generator deliberately builds in the structure the analysis is meant to
find: a Pareto revenue curve (for ABC), a wide spread of demand variability
(for XYZ), weekly seasonality plus trend (for forecasting), and shipping-mode
dependent late-delivery behaviour (for OTIF).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import RANDOM_SEED, SYNTH_DATA_FILE

# Scheduled transit days by shipping mode -- matches DataCo's contract.
SHIPPING_MODES = {
    "Standard Class": {"share": 0.60, "scheduled": 4},
    "Second Class":   {"share": 0.20, "scheduled": 2},
    "First Class":    {"share": 0.15, "scheduled": 1},
    "Same Day":       {"share": 0.05, "scheduled": 0},
}

# Probability an order on a given mode misses its scheduled date.
LATE_PROB = {
    "Standard Class": 0.62,
    "Second Class":   0.55,
    "First Class":    0.48,
    "Same Day":       0.10,
}

REGIONS = {
    "Western Europe":   {"share": 0.22, "late_mult": 0.85},
    "Central America":  {"share": 0.19, "late_mult": 1.10},
    "South America":    {"share": 0.13, "late_mult": 1.15},
    "Northern Europe":  {"share": 0.12, "late_mult": 0.80},
    "Southeast Asia":   {"share": 0.11, "late_mult": 1.20},
    "US Center":        {"share": 0.10, "late_mult": 0.90},
    "Southern Europe":  {"share": 0.08, "late_mult": 0.95},
    "West Africa":      {"share": 0.05, "late_mult": 1.30},
}

CATEGORIES = {
    "Beverages":        {"n_sku": 24, "price": (2.0, 9.0),   "margin": 0.14},
    "Snacks":           {"n_sku": 24, "price": (1.5, 7.0),   "margin": 0.18},
    "Personal Care":    {"n_sku": 21, "price": (3.0, 22.0),  "margin": 0.26},
    "Home Care":        {"n_sku": 19, "price": (4.0, 18.0),  "margin": 0.21},
    "Dairy":            {"n_sku": 17, "price": (1.2, 6.5),   "margin": 0.10},
    "Packaged Foods":   {"n_sku": 15, "price": (2.5, 14.0),  "margin": 0.16},
}

# Fraction of the catalogue that behaves as slow-moving / lumpy demand:
# long runs of zero weeks broken by occasional bulk orders. These become the
# Z class in XYZ and are the SKUs a planner most needs flagged.
INTERMITTENT_SHARE = 0.30

# High-revenue but campaign-driven SKUs -> the AY / AZ cells.
PROMO_SHARE = 0.15

SEGMENTS = {"Consumer": 0.52, "Corporate": 0.31, "Home Office": 0.17}


def _build_sku_master(rng: np.random.Generator) -> pd.DataFrame:
    """One row per SKU: category, unit price, base daily demand, volatility."""
    rows = []
    sku_id = 1
    for cat, spec in CATEGORIES.items():
        for j in range(spec["n_sku"]):
            lo, hi = spec["price"]
            rows.append(
                {
                    "product_id": f"SKU-{sku_id:04d}",
                    "product_name": f"{cat.split()[0]} Item {j + 1:02d}",
                    "category_name": cat,
                    "unit_price": round(float(rng.uniform(lo, hi)), 2),
                    "base_margin": spec["margin"],
                }
            )
            sku_id += 1
    sku = pd.DataFrame(rows)
    n = len(sku)

    # Lognormal base demand -> a natural Pareto revenue curve for ABC.
    sku["base_daily_orders"] = rng.lognormal(mean=0.00, sigma=0.95, size=n)
    sku["volatility"] = rng.uniform(0.15, 0.75, size=n)
    sku["profile"] = "steady"

    pool = rng.permutation(n)

    # Slow movers: a small base rate combined with very high overdispersion.
    # In a gamma-Poisson mixture this yields mostly-zero weeks punctuated by
    # spikes -- the lumpy signature that pushes weekly CV above 1.0. These
    # land in the C rows and the Z column.
    n_int = int(round(INTERMITTENT_SHARE * n))
    idx_int = pool[:n_int]
    sku.loc[idx_int, "profile"] = "intermittent"
    sku.loc[idx_int, "base_daily_orders"] *= rng.uniform(0.02, 0.10, size=n_int)
    sku.loc[idx_int, "volatility"] = rng.uniform(2.0, 5.0, size=n_int)

    # Promo / seasonal lines: high revenue but erratic, driven by campaigns
    # rather than baseline consumption. These populate the AY and AZ cells,
    # which are the commercially interesting ones -- high value AND hard to
    # forecast, so they carry the most inventory risk.
    n_promo = int(round(PROMO_SHARE * n))
    idx_promo = pool[n_int:n_int + n_promo]
    sku.loc[idx_promo, "profile"] = "promo"
    sku.loc[idx_promo, "base_daily_orders"] *= rng.uniform(1.6, 4.0, size=n_promo)
    sku.loc[idx_promo, "volatility"] = rng.uniform(1.1, 2.4, size=n_promo)

    # Per-SKU weekly seasonality strength and linear trend.
    sku["season_amp"] = rng.uniform(0.05, 0.40, size=n)
    sku["trend"] = rng.uniform(-0.00020, 0.00055, size=n)
    return sku


def generate(n_days: int = 1096, start: str = "2015-01-01",
             seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sku = _build_sku_master(rng)
    dates = pd.date_range(start=start, periods=n_days, freq="D")

    # ---- SKU x date grid, then draw an order count for every cell ----------
    grid = sku.merge(pd.DataFrame({"order_date": dates}), how="cross")
    day_idx = (grid["order_date"] - dates[0]).dt.days.to_numpy()
    dow = grid["order_date"].dt.dayofweek.to_numpy()

    seasonal = 1.0 + grid["season_amp"].to_numpy() * np.sin(2 * np.pi * dow / 7.0)
    trend = 1.0 + grid["trend"].to_numpy() * day_idx
    # December lift, typical FMCG festive peak.
    festive = np.where(grid["order_date"].dt.month.to_numpy() == 12, 1.25, 1.0)

    lam = grid["base_daily_orders"].to_numpy() * seasonal * trend * festive
    lam = np.clip(lam, 0.01, None)

    # Gamma-Poisson mixture = negative binomial: overdispersion controlled by
    # per-SKU volatility, which is exactly what XYZ later measures.
    shape = 1.0 / np.clip(grid["volatility"].to_numpy(), 0.05, None) ** 2
    lam_noisy = rng.gamma(shape=shape, scale=lam / shape)
    n_orders = rng.poisson(lam_noisy)

    grid = grid.loc[n_orders > 0].copy()
    counts = n_orders[n_orders > 0]
    lines = grid.loc[grid.index.repeat(counts)].reset_index(drop=True)
    n = len(lines)

    # ---- order attributes --------------------------------------------------
    lines["order_item_quantity"] = rng.integers(1, 6, size=n)
    disc = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20],
                      size=n, p=[0.42, 0.23, 0.18, 0.11, 0.06])
    lines["order_item_discount_rate"] = disc
    gross = lines["unit_price"].to_numpy() * lines["order_item_quantity"].to_numpy()
    lines["sales"] = np.round(gross * (1 - disc), 2)

    margin_noise = rng.normal(0, 0.09, size=n)
    lines["order_profit_per_order"] = np.round(
        lines["sales"].to_numpy() * (lines["base_margin"].to_numpy() + margin_noise), 2
    )

    modes = list(SHIPPING_MODES)
    lines["shipping_mode"] = rng.choice(
        modes, size=n, p=[SHIPPING_MODES[m]["share"] for m in modes]
    )
    regions = list(REGIONS)
    lines["order_region"] = rng.choice(
        regions, size=n, p=[REGIONS[r]["share"] for r in regions]
    )
    segs = list(SEGMENTS)
    lines["customer_segment"] = rng.choice(
        segs, size=n, p=[SEGMENTS[s] for s in segs]
    )

    # ---- shipping performance ---------------------------------------------
    sched = lines["shipping_mode"].map(
        {m: SHIPPING_MODES[m]["scheduled"] for m in modes}
    ).to_numpy()
    lines["days_for_shipment_scheduled"] = sched

    base_late = lines["shipping_mode"].map(LATE_PROB).to_numpy()
    mult = lines["order_region"].map({r: REGIONS[r]["late_mult"] for r in regions}).to_numpy()
    # Q4 congestion makes delivery worse exactly when volume peaks.
    q4 = np.where(lines["order_date"].dt.quarter.to_numpy() == 4, 1.12, 1.0)
    p_late = np.clip(base_late * mult * q4, 0.02, 0.95)

    is_late = rng.random(n) < p_late
    slip = rng.choice([1, 2, 3, 4], size=n, p=[0.50, 0.28, 0.15, 0.07])
    early = rng.choice([0, 1, 2], size=n, p=[0.55, 0.32, 0.13])
    real = np.where(is_late, sched + slip, np.maximum(sched - early, 0))
    lines["days_for_shipping_real"] = real
    lines["late_delivery_risk"] = is_late.astype(int)

    status = np.where(real > sched, "Late delivery",
                      np.where(real < sched, "Advance shipping", "Shipping on time"))
    # A small slice of orders is cancelled -- these must be excluded from OTIF.
    cancelled = rng.random(n) < 0.023
    status = np.where(cancelled, "Shipping canceled", status)
    lines["delivery_status"] = status
    lines["order_status"] = np.where(cancelled, "CANCELED", "COMPLETE")

    lines["order_id"] = np.arange(1, n + 1)
    lines["order_item_id"] = np.arange(1, n + 1)

    out_cols = [
        "order_id", "order_item_id", "order_date", "product_id", "product_name",
        "category_name", "customer_segment", "order_region", "shipping_mode",
        "days_for_shipment_scheduled", "days_for_shipping_real",
        "delivery_status", "late_delivery_risk", "order_status",
        "order_item_quantity", "unit_price", "order_item_discount_rate",
        "sales", "order_profit_per_order",
    ]
    return lines[out_cols].sort_values("order_date").reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv(SYNTH_DATA_FILE, index=False)
    print(f"Wrote {len(df):,} order lines -> {SYNTH_DATA_FILE}")
    print(f"Date range : {df.order_date.min().date()} to {df.order_date.max().date()}")
    print(f"SKUs       : {df.product_id.nunique()}")
    print(f"Late rate  : {df.late_delivery_risk.mean():.1%}")
