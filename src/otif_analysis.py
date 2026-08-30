"""
Module A -- OTIF and delivery performance.

OTIF ("On Time In Full") is the standard service KPI in supply chain. An order
line counts as a success only if it was actually fulfilled AND arrived no later
than the promised date:

    on_time = (days_for_shipping_real <= days_for_shipment_scheduled)

The headline number is not the point. The point is WHERE the failures
concentrate: this module ranks shipping modes and regions by their
*contribution* to the total late count, which is what tells a planner where to
intervene. A segment with a terrible rate but tiny volume is not the problem.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

from config import FIGURES, TABLES

plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3})


def overall_kpis(df: pd.DataFrame) -> dict:
    f = df.loc[df["is_fulfilled"]]
    return {
        "order_lines_total": int(len(df)),
        "order_lines_fulfilled": int(len(f)),
        "cancelled_or_fraud": int((~df["is_fulfilled"]).sum()),
        "on_time_rate": float(f["on_time"].mean()),
        "late_rate": float(f["is_late"].mean()),
        "avg_slip_days_when_late": float(
            f.loc[f["is_late"], "shipping_slip_days"].mean()
        ),
        "avg_transit_days": float(f["days_for_shipping_real"].mean()),
        "total_revenue": float(f["revenue"].sum()),
        "total_profit": float(f["profit"].sum()),
        "revenue_on_late_orders": float(f.loc[f["is_late"], "revenue"].sum()),
    }


def breakdown(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    """
    Late-rate table for one dimension, plus each level's SHARE OF ALL LATE
    ORDERS. `late_share` is the column that drives the recommendation, because
    it weights the failure rate by volume.
    """
    f = df.loc[df["is_fulfilled"]]
    total_late = f["is_late"].sum()

    g = (
        f.groupby(dim)
        .agg(lines=("is_late", "size"),
             late=("is_late", "sum"),
             avg_slip=("shipping_slip_days", "mean"),
             revenue=("revenue", "sum"))
        .assign(
            late_rate=lambda x: x["late"] / x["lines"],
            volume_share=lambda x: x["lines"] / len(f),
            late_share=lambda x: x["late"] / total_late,
        )
        .sort_values("late_share", ascending=False)
    )
    return g.reset_index()


def chi_square_test(df: pd.DataFrame, dim: str) -> dict:
    """
    Test whether late delivery is independent of `dim`.

    H0: late/on-time outcome is independent of the segment.
    A small p-value means the differences between segments are not explainable
    by sampling noise, which is what justifies acting on them.
    """
    f = df.loc[df["is_fulfilled"]]
    table = pd.crosstab(f[dim], f["is_late"])
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    # Cramer's V: effect size, 0 = no association, 1 = perfect. Needed because
    # with n>100k almost any difference is "significant".
    min_dim = min(table.shape) - 1
    cramers_v = float((chi2 / (n * min_dim)) ** 0.5) if min_dim > 0 else float("nan")
    return {"dimension": dim, "chi2": float(chi2), "p_value": float(p),
            "dof": int(dof), "cramers_v": cramers_v}


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    f = df.loc[df["is_fulfilled"]]
    return (
        f.groupby("year_month")
        .agg(lines=("is_late", "size"), late_rate=("is_late", "mean"),
             revenue=("revenue", "sum"))
        .reset_index()
    )


def plot_all(by_mode: pd.DataFrame, by_region: pd.DataFrame,
             by_cat: pd.DataFrame, trend: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    d = by_mode.sort_values("late_rate")
    a.barh(d["shipping_mode"], d["late_rate"], color="#c44e52")
    for i, (r, s) in enumerate(zip(d["late_rate"], d["volume_share"])):
        a.text(r + 0.01, i, f"{r:.0%}  (vol {s:.0%})", va="center", fontsize=8)
    a.set_xlim(0, 1.0)
    a.set_title("Late-delivery rate by shipping mode")
    a.set_xlabel("share of fulfilled lines delivered late")

    a = ax[0, 1]
    d = by_region.sort_values("late_share")
    a.barh(d["order_region"], d["late_share"], color="#4c72b0")
    a.set_title("Share of ALL late orders, by region")
    a.set_xlabel("contribution to total late count")

    a = ax[1, 0]
    d = by_cat.sort_values("late_rate")
    a.barh(d["category_name"], d["late_rate"], color="#55a868")
    a.set_xlim(0, 1.0)
    a.set_title("Late-delivery rate by category")
    a.set_xlabel("late rate")

    a = ax[1, 1]
    a.plot(range(len(trend)), trend["late_rate"], marker="o", ms=3,
           color="#8172b2")
    step = max(1, len(trend) // 8)
    a.set_xticks(range(0, len(trend), step))
    a.set_xticklabels(trend["year_month"][::step], rotation=45, ha="right",
                      fontsize=8)
    a.set_title("Late-delivery rate over time")
    a.set_ylabel("late rate")

    fig.suptitle("OTIF / delivery performance diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_otif_diagnostics.png")
    plt.close(fig)


def run(df: pd.DataFrame) -> dict:
    kpis = overall_kpis(df)
    by_mode = breakdown(df, "shipping_mode")
    by_region = breakdown(df, "order_region")
    by_cat = breakdown(df, "category_name")
    trend = monthly_trend(df)

    tests = [chi_square_test(df, d)
             for d in ["shipping_mode", "order_region", "category_name"]]

    by_mode.to_csv(TABLES / "otif_by_shipping_mode.csv", index=False)
    by_region.to_csv(TABLES / "otif_by_region.csv", index=False)
    by_cat.to_csv(TABLES / "otif_by_category.csv", index=False)
    trend.to_csv(TABLES / "otif_monthly_trend.csv", index=False)
    pd.DataFrame(tests).to_csv(TABLES / "otif_chi_square_tests.csv", index=False)

    plot_all(by_mode, by_region, by_cat, trend)

    # Concentration headline: how much of the late problem sits in the worst
    # mode plus the worst two regions.
    top_mode = by_mode.iloc[0]
    top2_regions = by_region.head(2)
    kpis["worst_mode"] = str(top_mode["shipping_mode"])
    kpis["worst_mode_late_share"] = float(top_mode["late_share"])
    kpis["worst_mode_late_rate"] = float(top_mode["late_rate"])
    kpis["top2_region_late_share"] = float(top2_regions["late_share"].sum())
    kpis["tests"] = tests
    return {"kpis": kpis, "by_mode": by_mode, "by_region": by_region,
            "by_category": by_cat, "trend": trend}


if __name__ == "__main__":
    orders = pd.read_csv("../data/processed/orders_clean.csv",
                         parse_dates=["order_date"])
    res = run(orders)
    k = res["kpis"]
    print(f"Fulfilled lines      : {k['order_lines_fulfilled']:,}")
    print(f"On-time rate         : {k['on_time_rate']:.1%}")
    print(f"Late rate            : {k['late_rate']:.1%}")
    print(f"Avg slip when late   : {k['avg_slip_days_when_late']:.2f} days")
    print(f"Worst mode           : {k['worst_mode']} "
          f"(rate {k['worst_mode_late_rate']:.1%}, "
          f"{k['worst_mode_late_share']:.1%} of all late orders)")
    print(f"Revenue on late lines: {k['revenue_on_late_orders']:,.0f}")
    print("\n", res["by_mode"].to_string(index=False))
