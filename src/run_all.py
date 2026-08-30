"""
End-to-end pipeline runner.

    python src/run_all.py

Executes data prep -> OTIF -> ABC-XYZ -> forecasting -> inventory, writes every
table to outputs/tables, every chart to outputs/figures, and a consolidated
outputs/RESULTS.md plus outputs/results.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import abc_xyz
import data_prep
import forecasting
import inventory
import otif_analysis
from config import ROOT, SYNTH_DATA_FILE, REAL_DATA_FILE


def _ensure_input() -> None:
    if not REAL_DATA_FILE.exists() and not SYNTH_DATA_FILE.exists():
        print("[0/5] No input found -> generating synthetic dataset")
        import make_synthetic_data
        make_synthetic_data.generate().to_csv(SYNTH_DATA_FILE, index=False)


def main() -> dict:
    _ensure_input()

    print("[1/5] Data preparation")
    orders, daily, audit = data_prep.run()
    print(f"      source={audit['source']}  clean lines={audit['rows_out']:,}")

    print("[2/5] OTIF / delivery performance")
    otif = otif_analysis.run(orders)
    k = otif["kpis"]
    print(f"      late rate={k['late_rate']:.1%}  "
          f"worst mode={k['worst_mode']} ({k['worst_mode_late_share']:.1%} of late)")

    print("[3/5] ABC-XYZ segmentation")
    seg = abc_xyz.run(daily)
    h = seg["headline"]
    print(f"      {h['n_A']} A-class SKUs = {h['A_revenue_share']:.0%} of revenue; "
          f"{h['n_Z']} erratic (Z) SKUs")

    print("[4/5] Demand forecasting (rolling-origin CV)")
    focus = (seg["segments"].loc[seg["segments"]["abc"].isin(["A", "B"]),
                                 "product_id"].astype(str).tolist())
    fc = forecasting.run(daily, sku_subset=focus)
    fh = fc["headline"]
    print(f"      best={fh['best_model']}  WMAPE={fh['best_wmape']:.3f}  "
          f"({fh['improvement_pct']:.1f}% vs seasonal naive)")

    print("[5/5] Inventory optimisation")
    prices = (orders.assign(product_id=orders["product_id"].astype(str))
              .groupby("product_id", as_index=False)["unit_price"].median())
    segs = seg["segments"].copy()
    segs["product_id"] = segs["product_id"].astype(str)
    inv = inventory.run(daily, segs, prices)
    ih = inv["headline"]
    print(f"      inventory capital {ih['inventory_reduction_pct']:.1f}% lower, "
          f"fill rate {ih['opt_fill_rate']:.2%}")

    results = {
        "data_source": audit["source"],
        "data_audit": {k2: v for k2, v in audit.items() if k2 != "source"},
        "otif": {k2: v for k2, v in k.items() if k2 != "tests"},
        "otif_tests": k["tests"],
        "segmentation": h,
        "forecasting": fh,
        "inventory": ih,
    }
    out = ROOT / "outputs"
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    _write_markdown(results, otif, seg, fc, inv, out / "RESULTS.md")
    print(f"\nDone. See {out/'RESULTS.md'}")
    return results


def _write_markdown(res, otif, seg, fc, inv, path: Path) -> None:
    k, h, fh, ih = res["otif"], res["segmentation"], res["forecasting"], res["inventory"]
    L = []
    L.append("# Results\n")
    L.append(f"Data source: `{res['data_source']}`  \n")
    L.append(f"Clean order lines: **{k['order_lines_total']:,}** "
             f"({k['order_lines_fulfilled']:,} fulfilled, "
             f"{k['cancelled_or_fraud']:,} cancelled/fraud excluded from OTIF)\n")

    L.append("\n## 1. Delivery performance (OTIF)\n")
    L.append(f"- On-time rate: **{k['on_time_rate']:.1%}** "
             f"(late rate {k['late_rate']:.1%})")
    L.append(f"- Average slip when late: **{k['avg_slip_days_when_late']:.2f} days**")
    L.append(f"- Worst mode: **{k['worst_mode']}** at a "
             f"{k['worst_mode_late_rate']:.1%} late rate, carrying "
             f"**{k['worst_mode_late_share']:.1%} of all late orders**")
    L.append(f"- Top 2 regions account for "
             f"**{k['top2_region_late_share']:.1%}** of late orders")
    L.append(f"- Revenue exposed to late delivery: "
             f"**{k['revenue_on_late_orders']:,.0f}**\n")
    L.append("\n### Chi-square independence tests\n")
    L.append("| Dimension | chi2 | p-value | Cramer's V |")
    L.append("|---|---:|---:|---:|")
    for t in res["otif_tests"]:
        L.append(f"| {t['dimension']} | {t['chi2']:,.1f} | "
                 f"{t['p_value']:.3g} | {t['cramers_v']:.3f} |")
    L.append("\n![OTIF](figures/01_otif_diagnostics.png)\n")

    L.append("\n## 2. ABC-XYZ segmentation\n")
    L.append(f"- **{h['n_A']}** A-class SKUs ({h['A_sku_share']:.0%} of the "
             f"catalogue) drive **{h['A_revenue_share']:.0%}** of revenue")
    L.append(f"- **{h['n_Z']}** SKUs are erratic (weekly CV >= 1.0)")
    L.append(f"- **{h['n_AX']}** SKUs qualify for automated replenishment (AX)\n")
    L.append("\n| Segment | SKUs | Revenue share | Mean CV | Policy |")
    L.append("|---|---:|---:|---:|---|")
    for _, r in seg["summary"].iterrows():
        L.append(f"| {r['segment']} | {int(r['n_skus'])} | "
                 f"{r['revenue_share']:.1%} | {r['avg_cv']:.2f} | {r['policy']} |")
    L.append("\n![ABC-XYZ](figures/02_abc_xyz.png)\n")

    L.append("\n## 3. Demand forecasting\n")
    L.append(f"Rolling-origin CV, {fh['n_folds']} folds x 28-day horizon, "
             f"{fh['n_skus']} A/B SKUs.\n")
    L.append("| Model | WMAPE | MAE | RMSE | Bias | vs seasonal naive |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for _, r in fc["summary"].iterrows():
        L.append(f"| {r['model']} | {r['wmape']:.3f} | {r['mae']:.2f} | "
                 f"{r['rmse']:.2f} | {r['bias']:+.2f} | "
                 f"{r['vs_seasonal_naive_pct']:+.1f}% |")
    L.append(f"\nBest model: **{fh['best_model']}**, WMAPE "
             f"**{fh['best_wmape']:.3f}**, **{fh['improvement_pct']:.1f}%** "
             f"better than the seasonal-naive baseline.\n")
    L.append("\n![Forecast](figures/03_forecast_performance.png)\n")

    L.append("\n## 4. Inventory optimisation\n")
    L.append("| Policy | Fill rate | Avg inventory value | Holding cost | "
             "Ordering cost | Total cost |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for _, r in inv["comparison"].iterrows():
        L.append(f"| {r['policy']} | {r['fill_rate']:.2%} | "
                 f"{r['avg_inventory_value']:,.0f} | "
                 f"{r['annual_holding_cost']:,.0f} | "
                 f"{r['ordering_cost']:,.0f} | {r['total_cost']:,.0f} |")
    L.append(f"\nSegment-differentiated safety stock cuts inventory capital by "
             f"**{ih['inventory_reduction_pct']:.1f}%** and total cost by "
             f"**{ih['total_cost_reduction_pct']:.1f}%** versus a flat 21-day "
             f"buffer, while holding fill rate at **{ih['opt_fill_rate']:.2%}**.")
    L.append(f"Against a uniform 99% rule it still saves "
             f"**{ih['vs_uniform_inventory_pct']:.1f}%** of capital, which shows "
             f"the gain comes from differentiating service levels rather than "
             f"simply holding less stock.\n")
    L.append("\n![Inventory](figures/04_inventory_optimisation.png)\n")

    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
