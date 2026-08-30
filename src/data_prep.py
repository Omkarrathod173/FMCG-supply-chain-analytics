"""
Load, normalise and clean the supply-chain order-line data.

Handles two sources transparently:
  1. The real Kaggle DataCo file (data/raw/DataCoSupplyChainDataset.csv)
  2. The synthetic fallback (data/raw/synthetic_supply_chain.csv)

The real file uses long, space-and-parenthesis column names and latin-1
encoding. REAL_COLUMN_MAP renames it onto the same snake_case schema the
synthetic generator emits, so every downstream module is source-agnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, REAL_DATA_FILE, SYNTH_DATA_FILE

# Real DataCo header -> canonical snake_case name used throughout the project.
REAL_COLUMN_MAP = {
    "Order Id": "order_id",
    "Order Item Id": "order_item_id",
    "order date (DateOrders)": "order_date",
    "Product Card Id": "product_id",
    "Product Name": "product_name",
    "Category Name": "category_name",
    "Customer Segment": "customer_segment",
    "Order Region": "order_region",
    "Shipping Mode": "shipping_mode",
    "Days for shipment (scheduled)": "days_for_shipment_scheduled",
    "Days for shipping (real)": "days_for_shipping_real",
    "Delivery Status": "delivery_status",
    "Late_delivery_risk": "late_delivery_risk",
    "Order Status": "order_status",
    "Order Item Quantity": "order_item_quantity",
    "Product Price": "unit_price",
    "Order Item Discount Rate": "order_item_discount_rate",
    "Sales": "sales",
    "Order Profit Per Order": "order_profit_per_order",
}

REQUIRED = [
    "order_id", "order_date", "product_id", "category_name", "order_region",
    "shipping_mode", "days_for_shipment_scheduled", "days_for_shipping_real",
    "order_status", "order_item_quantity", "sales", "order_profit_per_order",
]


def load_raw() -> tuple[pd.DataFrame, str]:
    """Return (raw dataframe, source label). Prefers the real file if present."""
    if REAL_DATA_FILE.exists():
        # The Kaggle export is latin-1, not utf-8. Reading it as utf-8 raises.
        df = pd.read_csv(REAL_DATA_FILE, encoding="latin-1", low_memory=False)
        df = df.rename(columns=REAL_COLUMN_MAP)
        source = "real:DataCoSupplyChainDataset"
    elif SYNTH_DATA_FILE.exists():
        df = pd.read_csv(SYNTH_DATA_FILE)
        source = "synthetic"
    else:
        raise FileNotFoundError(
            "No input data. Either place DataCoSupplyChainDataset.csv in "
            "data/raw/, or run `python src/make_synthetic_data.py` first."
        )

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")
    return df, source


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the order lines and record what was removed, for the audit trail."""
    audit = {"rows_in": len(df)}

    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Drop rows with no usable date -- they cannot join any time series.
    df = df.loc[df["order_date"].notna()]
    audit["dropped_bad_date"] = audit["rows_in"] - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["order_id", "product_id"], keep="first")
    audit["dropped_duplicate_lines"] = before - len(df)

    # Coerce numerics; non-numeric junk becomes NaN rather than crashing later.
    num_cols = [
        "days_for_shipment_scheduled", "days_for_shipping_real",
        "order_item_quantity", "sales", "order_profit_per_order",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.loc[df[num_cols].notna().all(axis=1)]
    audit["dropped_non_numeric"] = before - len(df)

    before = len(df)
    df = df.loc[(df["order_item_quantity"] > 0) & (df["sales"] >= 0)]
    audit["dropped_invalid_qty_or_sales"] = before - len(df)

    df["product_id"] = df["product_id"].astype(str)
    df["order_status"] = df["order_status"].astype(str).str.upper().str.strip()

    audit["rows_out"] = len(df)
    return df.reset_index(drop=True), audit


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the delivery-performance and calendar features the analysis needs."""
    df = df.copy()

    # Slip = actual minus promised. Positive means late, negative means early.
    df["shipping_slip_days"] = (
        df["days_for_shipping_real"] - df["days_for_shipment_scheduled"]
    )

    # Cancelled / suspected-fraud lines never physically shipped, so they are
    # not delivery failures. They are excluded from the OTIF denominator.
    df["is_fulfilled"] = ~df["order_status"].isin(
        ["CANCELED", "CANCELLED", "SUSPECTED_FRAUD"]
    )

    df["on_time"] = (df["shipping_slip_days"] <= 0) & df["is_fulfilled"]
    df["is_late"] = (df["shipping_slip_days"] > 0) & df["is_fulfilled"]

    df["revenue"] = df["sales"]
    df["profit"] = df["order_profit_per_order"]
    df["profit_margin"] = np.where(
        df["revenue"] > 0, df["profit"] / df["revenue"], np.nan
    )

    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["week"] = df["order_date"].dt.isocalendar().week.astype(int)
    return df


def build_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse order lines to one row per (product_id, date) with total units.

    Critical step: the grid is reindexed to a COMPLETE calendar per SKU and
    gaps filled with 0. Without this, a day with no orders is simply absent,
    and every rolling window and lag feature downstream silently shifts.
    """
    daily = (
        df.loc[df["is_fulfilled"]]
        .groupby(["product_id", "order_date"], as_index=False)
        .agg(units=("order_item_quantity", "sum"),
             revenue=("revenue", "sum"),
             orders=("order_id", "nunique"))
    )
    full_dates = pd.date_range(daily["order_date"].min(),
                               daily["order_date"].max(), freq="D")
    skus = daily["product_id"].unique()
    grid = pd.MultiIndex.from_product([skus, full_dates],
                                      names=["product_id", "order_date"])
    daily = (
        daily.set_index(["product_id", "order_date"])
        .reindex(grid, fill_value=0)
        .reset_index()
    )
    return daily


def run() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw, source = load_raw()
    clean_df, audit = clean(raw)
    feat = engineer(clean_df)
    daily = build_daily_demand(feat)

    audit["source"] = source
    feat.to_csv(DATA_PROCESSED / "orders_clean.csv", index=False)
    daily.to_csv(DATA_PROCESSED / "daily_demand.csv", index=False)
    return feat, daily, audit


if __name__ == "__main__":
    feat, daily, audit = run()
    print("--- data prep audit ---")
    for k, v in audit.items():
        print(f"{k:>28}: {v}")
    print(f"{'fulfilled lines':>28}: {feat.is_fulfilled.sum():,}")
    print(f"{'daily demand rows':>28}: {len(daily):,}")
