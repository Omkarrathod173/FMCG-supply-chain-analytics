# Power BI dashboard — build guide

Three pages built on the CSVs in `outputs/tables/`. Build it, screenshot each
page, and put the screenshots in the README — a dashboard nobody can see does
nothing for your resume.

## Data sources

| Table | File | Grain |
|---|---|---|
| `orders` | `data/processed/orders_clean.csv` | one row per order line |
| `otif_mode` | `outputs/tables/otif_by_shipping_mode.csv` | shipping mode |
| `otif_region` | `outputs/tables/otif_by_region.csv` | region |
| `segments` | `outputs/tables/abc_xyz_sku_level.csv` | SKU |
| `policy` | `outputs/tables/inventory_policy_by_sku.csv` | SKU |
| `curve` | `outputs/tables/service_level_cost_curve.csv` | service level |
| `forecast` | `outputs/tables/forecast_predictions.csv` | SKU × date |

## Model

Build a star schema, not a flat table. Create a date table and mark it as the
official date table (Modeling → Mark as date table), otherwise time
intelligence functions like `SAMEPERIODLASTYEAR` silently misbehave.

```
DimDate = CALENDAR(MIN(orders[order_date]), MAX(orders[order_date]))
```

Relationships: `DimDate[Date]` → `orders[order_date]` (one-to-many),
`segments[product_id]` → `orders[product_id]` (one-to-many),
`policy[product_id]` → `segments[product_id]` (one-to-one).

## Core DAX measures

```dax
Total Revenue = SUM(orders[sales])

Fulfilled Lines =
CALCULATE(COUNTROWS(orders), orders[is_fulfilled] = TRUE())

Late Lines =
CALCULATE(
    COUNTROWS(orders),
    orders[is_fulfilled] = TRUE(),
    orders[shipping_slip_days] > 0
)

OTIF % = DIVIDE([Fulfilled Lines] - [Late Lines], [Fulfilled Lines])

Avg Slip Days =
CALCULATE(
    AVERAGE(orders[shipping_slip_days]),
    orders[shipping_slip_days] > 0
)

-- Share of ALL late orders, ignoring the current filter on the visual.
-- ALLSELECTED respects slicers but ignores the row's own category filter,
-- which is what makes the denominator the visible total rather than the row.
Late Share % =
DIVIDE([Late Lines], CALCULATE([Late Lines], ALLSELECTED(orders)))

Revenue at Risk =
CALCULATE([Total Revenue], orders[shipping_slip_days] > 0)

OTIF vs Last Month =
[OTIF %] - CALCULATE([OTIF %], DATEADD(DimDate[Date], -1, MONTH))

Safety Stock Value = SUMX(policy, policy[ss_opt] * policy[unit_cost])
```

## Page 1 — Delivery performance

- KPI cards: `OTIF %`, `Avg Slip Days`, `Revenue at Risk`, `OTIF vs Last Month`
- Line chart: `OTIF %` by month (from `DimDate`)
- Bar chart: late rate by shipping mode, sorted descending
- Bar chart: `Late Share %` by region — this is the "where to act" visual
- Slicers: date range, category, region

Add a conditional-formatting rule on the OTIF card: red below 45%, amber to
55%, green above. A card with no threshold is just a number.

## Page 2 — SKU segmentation

- Matrix visual: ABC on rows, XYZ on columns, SKU count in values, with a
  colour scale on the cell background
- Scatter: revenue (log Y) vs coefficient of variation, coloured by XYZ class
- Table: SKU, segment, revenue, CV, recommended policy
- Drill-through page: single-SKU detail — demand history, current policy,
  reorder point, safety stock

Set the matrix cells to drill through to the SKU detail page. That interaction
is what separates a dashboard from a static report in a demo.

## Page 3 — Inventory and forecast

- Line chart: actual vs forecast demand over the held-out window (from `forecast`)
- Line chart: service level vs safety stock capital (from `curve`) — the
  convex curve is the most persuasive visual in the whole project
- Clustered bar: inventory value by policy (flat / uniform 99% / segmented)
- Bar: inventory capital by ABC-XYZ segment
- What-if parameter on target service level, driving a recalculated safety
  stock value

The what-if parameter is the piece worth demoing live. Modeling → New parameter
→ Numeric range, 0.80 to 0.99 in steps of 0.01, then write a measure that picks
the matching row from `curve`.

## Presentation notes

- Consistent colour semantics throughout: red always means late or at risk,
  green always means on time or optimised. Don't let the theme reassign them.
- Every page needs one sentence of takeaway text. "Standard Class carries 66.5%
  of all late orders" beats an unlabelled chart.
- Turn off visual interactions where cross-filtering confuses more than it
  helps (Format → Edit interactions).
