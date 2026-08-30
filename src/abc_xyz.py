"""
Module B -- ABC-XYZ segmentation.

ABC classifies SKUs by VALUE (Pareto):
    sort SKUs by revenue descending, take the running share of total revenue,
    A = up to 80%, B = up to 95%, C = the rest.
    Note the cut is on the CUMULATIVE share, not each SKU's own share.

XYZ classifies SKUs by PREDICTABILITY, using the coefficient of variation of
demand:
    CV = sigma / mu      (unitless, so a cheap high-volume SKU and an expensive
                          low-volume SKU are comparable)
    X = CV < 0.5   (steady, forecastable)
    Y = 0.5 <= CV < 1.0 (moderate, often seasonal)
    Z = CV >= 1.0  (erratic, lumpy)

Crossing them gives a 9-cell matrix that maps directly onto a stocking policy.
AX earns tight, automated replenishment with a thin buffer. CZ is not worth
actively planning. That mapping is the actual deliverable -- the classification
on its own is just a label.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ABC_CUTOFFS, FIGURES, TABLES, XYZ_CUTOFFS

# Policy recommendation for each of the 9 cells.
POLICY = {
    "AX": "Automate replenishment; low safety stock; tight cycle counting",
    "AY": "Weekly review; seasonal buffer; forecast with seasonality model",
    "AZ": "Manual planner review; high buffer or make-to-order; hedge supply",
    "BX": "Periodic auto-replenishment; standard buffer",
    "BY": "Monthly review; moderate buffer",
    "BZ": "Reduce commitment; consider vendor-managed inventory",
    "CX": "Bulk order infrequently; minimise handling cost",
    "CY": "Low priority; simple reorder point",
    "CZ": "Make-to-order or delist; do not hold stock",
}


def abc_classify(daily: pd.DataFrame) -> pd.DataFrame:
    """Revenue Pareto classification, one row per SKU."""
    rev = (
        daily.groupby("product_id", as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    rev["revenue_share"] = rev["revenue"] / rev["revenue"].sum()
    rev["cum_share"] = rev["revenue_share"].cumsum()

    a_cut, b_cut = ABC_CUTOFFS
    # pd.cut on the CUMULATIVE share. right=True means the boundary value
    # falls into the lower class, which is the convention planners expect.
    rev["abc"] = pd.cut(
        rev["cum_share"],
        bins=[0, a_cut, b_cut, 1.0001],
        labels=["A", "B", "C"],
        include_lowest=True,
    ).astype(str)
    rev["rank"] = np.arange(1, len(rev) + 1)
    return rev


def xyz_classify(daily: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Demand-variability classification.

    Aggregated to weekly buckets first. Daily FMCG demand has so many structural
    zeros that the daily CV mostly measures order frequency rather than genuine
    demand instability; weekly buckets give a CV that reflects planning risk at
    the horizon a planner actually operates on.
    """
    d = daily.copy()
    d["bucket"] = d["order_date"].dt.to_period(freq).dt.start_time
    per = d.groupby(["product_id", "bucket"], as_index=False)["units"].sum()

    stats = (
        per.groupby("product_id")["units"]
        .agg(mean_demand="mean", std_demand="std", n_periods="size")
        .reset_index()
    )
    stats["cv"] = np.where(
        stats["mean_demand"] > 0, stats["std_demand"] / stats["mean_demand"], np.inf
    )

    x_cut, y_cut = XYZ_CUTOFFS
    stats["xyz"] = np.select(
        [stats["cv"] < x_cut, stats["cv"] < y_cut],
        ["X", "Y"],
        default="Z",
    )
    return stats


def build_matrix(abc: pd.DataFrame, xyz: pd.DataFrame,
                 daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seg = abc.merge(xyz, on="product_id", how="inner")
    seg["segment"] = seg["abc"] + seg["xyz"]
    seg["policy"] = seg["segment"].map(POLICY)

    units = daily.groupby("product_id", as_index=False)["units"].sum()
    seg = seg.merge(units, on="product_id", how="left")

    summary = (
        seg.groupby("segment")
        .agg(n_skus=("product_id", "size"),
             revenue=("revenue", "sum"),
             units=("units", "sum"),
             avg_cv=("cv", "mean"))
        .assign(revenue_share=lambda x: x["revenue"] / x["revenue"].sum(),
                sku_share=lambda x: x["n_skus"] / x["n_skus"].sum())
        .sort_values("revenue", ascending=False)
        .reset_index()
    )
    summary["policy"] = summary["segment"].map(POLICY)
    return seg, summary


def plot_all(abc: pd.DataFrame, seg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    a = ax[0]
    a.plot(abc["rank"], abc["cum_share"], color="#4c72b0", lw=2)
    a.axhline(ABC_CUTOFFS[0], ls="--", c="#c44e52", lw=1)
    a.axhline(ABC_CUTOFFS[1], ls="--", c="#dd8452", lw=1)
    a.set_xlabel("SKU rank by revenue")
    a.set_ylabel("cumulative revenue share")
    a.set_title("Pareto curve (ABC cut points)")
    for lbl, y in zip(["A / B", "B / C"], ABC_CUTOFFS):
        a.text(len(abc) * 0.62, y + 0.015, lbl, fontsize=8, color="#444")

    a = ax[1]
    colors = {"X": "#55a868", "Y": "#dd8452", "Z": "#c44e52"}
    for cls, grp in seg.groupby("xyz"):
        a.scatter(grp["cv"], grp["revenue"], s=34, alpha=0.8,
                  label=cls, color=colors.get(cls, "#888"))
    for x in XYZ_CUTOFFS:
        a.axvline(x, ls="--", c="#666", lw=1)
    a.set_yscale("log")
    a.set_xlabel("coefficient of variation (weekly demand)")
    a.set_ylabel("revenue (log scale)")
    a.set_title("Value vs variability")
    a.legend(title="XYZ", fontsize=8)

    a = ax[2]
    pivot = (
        seg.pivot_table(index="abc", columns="xyz", values="product_id",
                        aggfunc="size")
        .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"])
        .fillna(0)
    )
    im = a.imshow(pivot.to_numpy(), cmap="Blues")
    a.set_xticks(range(3), ["X", "Y", "Z"])
    a.set_yticks(range(3), ["A", "B", "C"])
    for i in range(3):
        for j in range(3):
            v = int(pivot.to_numpy()[i, j])
            a.text(j, i, v, ha="center", va="center",
                   color="white" if v > pivot.to_numpy().max() / 2 else "black",
                   fontsize=12, fontweight="bold")
    a.set_title("ABC-XYZ matrix (SKU counts)")
    a.grid(False)
    fig.colorbar(im, ax=a, shrink=0.8)

    fig.tight_layout()
    fig.savefig(FIGURES / "02_abc_xyz.png")
    plt.close(fig)


def run(daily: pd.DataFrame) -> dict:
    abc = abc_classify(daily)
    xyz = xyz_classify(daily)
    seg, summary = build_matrix(abc, xyz, daily)

    seg.to_csv(TABLES / "abc_xyz_sku_level.csv", index=False)
    summary.to_csv(TABLES / "abc_xyz_summary.csv", index=False)
    plot_all(abc, seg)

    a_skus = seg.loc[seg["abc"] == "A"]
    headline = {
        "n_skus": int(len(seg)),
        "n_A": int((seg["abc"] == "A").sum()),
        "A_sku_share": float((seg["abc"] == "A").mean()),
        "A_revenue_share": float(a_skus["revenue"].sum() / seg["revenue"].sum()),
        "n_Z": int((seg["xyz"] == "Z").sum()),
        "n_AX": int((seg["segment"] == "AX").sum()),
        "n_AZ": int((seg["segment"] == "AZ").sum()),
        "AZ_revenue_share": float(
            seg.loc[seg["segment"] == "AZ", "revenue"].sum() / seg["revenue"].sum()
        ),
    }
    return {"abc": abc, "xyz": xyz, "segments": seg, "summary": summary,
            "headline": headline}


if __name__ == "__main__":
    daily = pd.read_csv("../data/processed/daily_demand.csv",
                        parse_dates=["order_date"])
    res = run(daily)
    h = res["headline"]
    print(f"SKUs                 : {h['n_skus']}")
    print(f"Class A SKUs         : {h['n_A']} ({h['A_sku_share']:.0%} of SKUs) "
          f"-> {h['A_revenue_share']:.0%} of revenue")
    print(f"Erratic (Z) SKUs     : {h['n_Z']}")
    print(f"AX (auto-replenish)  : {h['n_AX']}")
    print(f"AZ (planner review)  : {h['n_AZ']} "
          f"({h['AZ_revenue_share']:.0%} of revenue)")
    print("\n", res["summary"][["segment", "n_skus", "revenue_share",
                                "avg_cv"]].to_string(index=False))
