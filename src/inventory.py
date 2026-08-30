"""
Module D -- safety stock, reorder points and policy simulation.

CORE FORMULA (variable demand AND variable lead time):

    SS  = z * sqrt( L_bar * sigma_d^2  +  d_bar^2 * sigma_L^2 )
    ROP = d_bar * L_bar + SS

    d_bar, sigma_d : mean and std of DAILY demand
    L_bar, sigma_L : mean and std of lead time in DAYS
    z              : inverse normal CDF at the target cycle service level
                     (1.645 at 95%, 2.326 at 99%)

The two terms under the root are the two independent sources of risk: demand
varying while lead time is fixed, and lead time varying while demand runs at
its average. They add as variances, not as standard deviations, which is why
the square root sits outside the sum. Dropping the second term -- the common
shortcut SS = z*sigma_d*sqrt(L) -- assumes suppliers are perfectly reliable
and systematically under-buffers.

EOQ (economic order quantity):

    EOQ = sqrt( 2 * D * S / H )

    D = annual demand, S = fixed cost per order, H = annual holding cost/unit
It balances ordering cost against holding cost. The curve is flat near the
optimum, so being 20% off on EOQ costs very little -- worth knowing, because
it is a favourite interview follow-up.

WHY SIMULATE RATHER THAN TRUST THE FORMULA
The formula gives CYCLE service level: the probability of not stocking out in
a given replenishment cycle. What the business actually cares about is FILL
RATE: the fraction of demanded units actually served. They are not the same
number, and the gap widens for lumpy demand. The simulation measures fill rate
directly on held-out demand.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from config import (FIGURES, LEAD_TIME_MEAN_DAYS, LEAD_TIME_STD_DAYS,
                    RANDOM_SEED, SERVICE_LEVELS, TABLES,
                    TARGET_SERVICE_LEVEL, UNIT_HOLDING_COST_RATE)

ORDER_COST = 25.0        # fixed administrative cost per purchase order
COST_RATIO = 0.70         # unit cost modelled as 70% of selling price
FLAT_POLICY_DAYS = 21     # "current state": same days-of-cover for every SKU

# Differentiated targets. High-value SKUs earn a higher service level; the
# erratic C/Z tail is deliberately given less, because buffering lumpy demand
# to 98% ties up capital for almost no revenue protection.
SEGMENT_SERVICE = {
    "AX": 0.98, "AY": 0.97, "AZ": 0.95,
    "BX": 0.95, "BY": 0.95, "BZ": 0.92,
    "CX": 0.92, "CY": 0.90, "CZ": 0.85,
}


def demand_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of daily demand per SKU, plus annualised demand."""
    g = (
        daily.groupby("product_id")["units"]
        .agg(d_bar="mean", sigma_d="std", n_days="size", total_units="sum")
        .reset_index()
    )
    g["sigma_d"] = g["sigma_d"].fillna(0.0)
    g["annual_demand"] = g["d_bar"] * 365
    return g


def safety_stock(d_bar, sigma_d, z, L_mean=LEAD_TIME_MEAN_DAYS,
                 L_std=LEAD_TIME_STD_DAYS):
    """Safety stock under joint demand and lead-time uncertainty."""
    var = L_mean * np.square(sigma_d) + np.square(d_bar) * np.square(L_std)
    return z * np.sqrt(var)


def eoq(annual_demand, unit_cost, order_cost=ORDER_COST,
        holding_rate=UNIT_HOLDING_COST_RATE):
    H = np.maximum(holding_rate * unit_cost, 1e-6)
    return np.sqrt(2 * np.maximum(annual_demand, 0) * order_cost / H)


def build_policies(stats: pd.DataFrame, seg: pd.DataFrame,
                   prices: pd.DataFrame) -> pd.DataFrame:
    df = (
        stats.merge(seg[["product_id", "abc", "xyz", "segment"]], on="product_id")
        .merge(prices, on="product_id", how="left")
    )
    df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())
    df["unit_cost"] = df["unit_price"] * COST_RATIO

    # ---- policy 1: flat days-of-cover for every SKU (the naive baseline) ----
    df["ss_flat"] = df["d_bar"] * FLAT_POLICY_DAYS
    df["rop_flat"] = df["d_bar"] * LEAD_TIME_MEAN_DAYS + df["ss_flat"]

    # ---- policy 2: segment-differentiated statistical safety stock ---------
    df["target_sl"] = df["segment"].map(SEGMENT_SERVICE).fillna(TARGET_SERVICE_LEVEL)
    df["z"] = norm.ppf(df["target_sl"])
    df["ss_opt"] = safety_stock(df["d_bar"].to_numpy(), df["sigma_d"].to_numpy(),
                                df["z"].to_numpy())
    df["rop_opt"] = df["d_bar"] * LEAD_TIME_MEAN_DAYS + df["ss_opt"]

    # ---- policy 3: uniform 99% for every SKU (a common "just be safe" rule) -
    # Included to prove the saving comes from DIFFERENTIATING service levels,
    # not merely from holding less stock overall.
    z99 = norm.ppf(0.99)
    df["ss_uniform"] = safety_stock(df["d_bar"].to_numpy(),
                                    df["sigma_d"].to_numpy(), z99)
    df["rop_uniform"] = df["d_bar"] * LEAD_TIME_MEAN_DAYS + df["ss_uniform"]

    df["eoq"] = eoq(df["annual_demand"].to_numpy(), df["unit_cost"].to_numpy())
    # Never order less than roughly a week of demand -- avoids absurd tiny POs
    # on very slow movers where EOQ collapses toward zero.
    df["order_qty"] = np.maximum(df["eoq"], df["d_bar"] * 7).round()
    return df


def simulate_policy(daily: pd.DataFrame, policy: pd.DataFrame,
                    ss_col: str, rop_col: str, sim_days: int = 365,
                    seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Continuous-review (s, Q) simulation on held-out actual demand.

    Each day: receive any arriving order, serve demand from on-hand stock
    (unmet demand is lost, not backordered), then if the inventory position
    has fallen to or below the reorder point, place an order of Q units that
    arrives after a stochastic lead time.

    Inventory POSITION (on hand + on order) drives the reorder decision, not
    on-hand alone. Using on-hand would re-trigger an order every day while the
    first shipment is still in transit.
    """
    rng = np.random.default_rng(seed)
    last_date = daily["order_date"].max()
    window_start = last_date - pd.Timedelta(days=sim_days - 1)
    sim = daily.loc[daily["order_date"] >= window_start]

    rows = []
    for _, p in policy.iterrows():
        sku = p["product_id"]
        d = (sim.loc[sim["product_id"] == sku]
             .sort_values("order_date")["units"].to_numpy(dtype=float))
        if d.size == 0:
            continue

        rop, Q = float(p[rop_col]), float(p["order_qty"])
        on_hand = float(p[rop_col] + p["order_qty"])   # start fully stocked
        pipeline: dict[int, float] = {}
        on_order = 0.0

        demand_total = shortfall_total = 0.0
        oh_trace = np.empty(d.size)
        n_orders = stockout_days = 0

        for t, dem in enumerate(d):
            if t in pipeline:
                arrived = pipeline.pop(t)
                on_hand += arrived
                on_order -= arrived

            served = min(on_hand, dem)
            on_hand -= served
            short = dem - served
            demand_total += dem
            shortfall_total += short
            if short > 0:
                stockout_days += 1

            if on_hand + on_order <= rop:
                lt = int(max(1, round(rng.normal(LEAD_TIME_MEAN_DAYS,
                                                 LEAD_TIME_STD_DAYS))))
                pipeline[t + lt] = pipeline.get(t + lt, 0.0) + Q
                on_order += Q
                n_orders += 1

            oh_trace[t] = on_hand

        avg_oh = float(oh_trace.mean())
        rows.append({
            "product_id": sku,
            "segment": p["segment"],
            "unit_cost": p["unit_cost"],
            "avg_on_hand_units": avg_oh,
            "avg_inventory_value": avg_oh * p["unit_cost"],
            "annual_holding_cost": avg_oh * p["unit_cost"] * UNIT_HOLDING_COST_RATE,
            "ordering_cost": n_orders * ORDER_COST,
            "demand_units": demand_total,
            "shortfall_units": shortfall_total,
            "fill_rate": 1 - shortfall_total / demand_total if demand_total > 0 else 1.0,
            "stockout_days": stockout_days,
            "n_orders": n_orders,
            "safety_stock": p[ss_col],
        })
    return pd.DataFrame(rows)


def aggregate(sim: pd.DataFrame, label: str) -> dict:
    dem, short = sim["demand_units"].sum(), sim["shortfall_units"].sum()
    return {
        "policy": label,
        "fill_rate": float(1 - short / dem) if dem > 0 else np.nan,
        "avg_inventory_value": float(sim["avg_inventory_value"].sum()),
        "annual_holding_cost": float(sim["annual_holding_cost"].sum()),
        "ordering_cost": float(sim["ordering_cost"].sum()),
        "total_cost": float(sim["annual_holding_cost"].sum()
                            + sim["ordering_cost"].sum()),
        "shortfall_units": float(short),
        "stockout_days": int(sim["stockout_days"].sum()),
        "avg_safety_stock_units": float(sim["safety_stock"].mean()),
    }


def service_cost_curve(stats: pd.DataFrame, seg: pd.DataFrame,
                       prices: pd.DataFrame) -> pd.DataFrame:
    """Safety-stock capital required at each candidate service level."""
    base = (stats.merge(seg[["product_id", "segment"]], on="product_id")
            .merge(prices, on="product_id", how="left"))
    base["unit_price"] = base["unit_price"].fillna(base["unit_price"].median())
    base["unit_cost"] = base["unit_price"] * COST_RATIO

    rows = []
    for sl in SERVICE_LEVELS:
        z = norm.ppf(sl)
        ss = safety_stock(base["d_bar"].to_numpy(), base["sigma_d"].to_numpy(), z)
        value = float((ss * base["unit_cost"]).sum())
        rows.append({
            "service_level": sl, "z": float(z),
            "safety_stock_units": float(ss.sum()),
            "safety_stock_value": value,
            "annual_holding_cost": value * UNIT_HOLDING_COST_RATE,
        })
    out = pd.DataFrame(rows)
    ref = out.loc[out["service_level"] == 0.90, "safety_stock_value"].iloc[0]
    out["value_vs_90pct"] = out["safety_stock_value"] / ref
    return out


def _plot(curve: pd.DataFrame, comparison: pd.DataFrame,
          sim_opt: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    a = ax[0]
    a.plot(curve["service_level"] * 100, curve["safety_stock_value"],
           marker="o", color="#4c72b0")
    a.axvline(TARGET_SERVICE_LEVEL * 100, ls="--", c="#c44e52", lw=1)
    a.set_xlabel("cycle service level (%)")
    a.set_ylabel("safety stock capital")
    a.set_title("Service level vs inventory investment")
    for _, r in curve.iterrows():
        a.annotate(f"{r['service_level']:.0%}",
                   (r["service_level"] * 100, r["safety_stock_value"]),
                   textcoords="offset points", xytext=(0, 7), fontsize=7,
                   ha="center")

    a = ax[1]
    x = np.arange(len(comparison))
    a.bar(x - 0.2, comparison["avg_inventory_value"], width=0.4,
          label="avg inventory value", color="#4c72b0")
    a2 = a.twinx()
    a2.plot(x, comparison["fill_rate"] * 100, marker="o", color="#c44e52",
            lw=2, label="fill rate")
    a2.set_ylabel("fill rate (%)", color="#c44e52")
    a.set_xticks(x, comparison["policy"])
    a.set_ylabel("inventory value")
    a.set_title("Flat buffer vs segment-optimised")
    a.legend(loc="upper left", fontsize=8)

    a = ax[2]
    by_seg = (sim_opt.groupby("segment")
              .agg(fill=("fill_rate", "mean"),
                   value=("avg_inventory_value", "sum"))
              .reset_index().sort_values("value", ascending=True))
    a.barh(by_seg["segment"], by_seg["value"], color="#55a868")
    a.set_xlabel("avg inventory value held")
    a.set_title("Where the capital sits, by ABC-XYZ segment")

    fig.tight_layout()
    fig.savefig(FIGURES / "04_inventory_optimisation.png")
    plt.close(fig)


def run(daily: pd.DataFrame, seg: pd.DataFrame,
        prices: pd.DataFrame) -> dict:
    stats = demand_stats(daily)
    policy = build_policies(stats, seg, prices)
    curve = service_cost_curve(stats, seg, prices)

    sim_flat = simulate_policy(daily, policy, "ss_flat", "rop_flat")
    sim_uni = simulate_policy(daily, policy, "ss_uniform", "rop_uniform")
    sim_opt = simulate_policy(daily, policy, "ss_opt", "rop_opt")

    comparison = pd.DataFrame([
        aggregate(sim_flat, "Flat 21-day buffer"),
        aggregate(sim_uni, "Uniform 99%"),
        aggregate(sim_opt, "Segment-optimised"),
    ])

    policy.to_csv(TABLES / "inventory_policy_by_sku.csv", index=False)
    curve.to_csv(TABLES / "service_level_cost_curve.csv", index=False)
    comparison.to_csv(TABLES / "inventory_policy_comparison.csv", index=False)
    sim_opt.to_csv(TABLES / "inventory_simulation_optimised.csv", index=False)

    _plot(curve, comparison, sim_opt)

    flat, uni, opt = comparison.iloc[0], comparison.iloc[1], comparison.iloc[2]
    headline = {
        "flat_fill_rate": float(flat["fill_rate"]),
        "opt_fill_rate": float(opt["fill_rate"]),
        "fill_rate_delta_pp": float((opt["fill_rate"] - flat["fill_rate"]) * 100),
        "flat_inventory_value": float(flat["avg_inventory_value"]),
        "opt_inventory_value": float(opt["avg_inventory_value"]),
        "inventory_reduction_pct": float(
            (flat["avg_inventory_value"] - opt["avg_inventory_value"])
            / flat["avg_inventory_value"] * 100),
        "total_cost_reduction_pct": float(
            (flat["total_cost"] - opt["total_cost"]) / flat["total_cost"] * 100),
        "shortfall_reduction_pct": float(
            (flat["shortfall_units"] - opt["shortfall_units"])
            / max(flat["shortfall_units"], 1) * 100),
    }
    headline["uniform_fill_rate"] = float(uni["fill_rate"])
    headline["uniform_inventory_value"] = float(uni["avg_inventory_value"])
    headline["vs_uniform_inventory_pct"] = float(
        (uni["avg_inventory_value"] - opt["avg_inventory_value"])
        / uni["avg_inventory_value"] * 100)
    return {"policy": policy, "curve": curve, "comparison": comparison,
            "sim_flat": sim_flat, "sim_uniform": sim_uni, "sim_opt": sim_opt,
            "headline": headline}


if __name__ == "__main__":
    daily = pd.read_csv("../data/processed/daily_demand.csv",
                        parse_dates=["order_date"])
    seg = pd.read_csv("../outputs/tables/abc_xyz_sku_level.csv")
    seg["product_id"] = seg["product_id"].astype(str)
    orders = pd.read_csv("../data/processed/orders_clean.csv",
                         usecols=["product_id", "unit_price"])
    orders["product_id"] = orders["product_id"].astype(str)
    prices = orders.groupby("product_id", as_index=False)["unit_price"].median()

    res = run(daily, seg, prices)
    print(res["comparison"].to_string(index=False))
    h = res["headline"]
    print(f"\nFill rate  : {h['flat_fill_rate']:.2%} -> {h['opt_fill_rate']:.2%} "
          f"({h['fill_rate_delta_pp']:+.2f} pp)")
    print(f"Inventory  : {h['inventory_reduction_pct']:.1f}% lower capital")
    print(f"Total cost : {h['total_cost_reduction_pct']:.1f}% lower")
