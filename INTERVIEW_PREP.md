# Interview prep — FMCG Supply Chain & Demand Analytics

Direct answers, not prompts. If you can say these out loud in your own words,
the project will hold up under a 45-minute technical drill.

---

## Part 1 — The 60-second project pitch

> I analysed ~237K FMCG order lines end to end. I found 57.8% of fulfilled
> lines arrive late, and rather than stopping at that number I decomposed it:
> Standard Class shipping carries 66.5% of all late orders, and chi-square
> tests showed late delivery is statistically independent of product category
> (p = 0.32) but strongly associated with shipping mode and region. So it's a
> logistics problem, not a product problem.
>
> Then I segmented 120 SKUs with ABC-XYZ — 21% of SKUs drive 79% of revenue —
> forecast demand with rolling-origin validation where Holt-Winters beat a
> seasonal-naive baseline by 24.6% on WMAPE, and used those forecasts to set
> safety stock and reorder points. Simulating the policy against held-out
> demand, segment-differentiated buffers cut inventory capital 34.6% versus a
> flat 21-day rule while holding fill rate above 99%.

---

## Part 2 — Terms you must be able to define cold

| Term | Definition |
|---|---|
| **OTIF** | On Time In Full. An order succeeds only if delivered by the promised date *and* complete. Two failure modes in one KPI. |
| **Fill rate** | Fraction of demanded **units** actually served from stock. |
| **Cycle service level** | Probability of not stocking out during a single replenishment cycle. Not the same as fill rate. |
| **Safety stock** | Buffer inventory held to absorb demand and lead-time variability. |
| **Reorder point (ROP)** | Inventory position that triggers a new order. Covers expected demand over the lead time plus safety stock. |
| **Lead time** | Days between placing a replenishment order and receiving it. |
| **EOQ** | Economic order quantity — the order size minimising ordering cost plus holding cost combined. |
| **Coefficient of variation (CV)** | sigma/mu. Unitless, so variability is comparable across SKUs of different scale. |
| **WMAPE** | Weighted Mean Absolute Percentage Error = sum\|y − yhat\| / sum\|y\|. |
| **Bullwhip effect** | Demand variability amplifying as it moves upstream from consumer to supplier. |
| **Bias (forecast)** | Mean signed error. Persistent non-zero bias means systematic over- or under-forecasting. |
| **Cramer's V** | Effect size for chi-square, 0 to 1. Says how *strong* an association is, where the p-value only says whether it exists. |
| **Croston's method** | Forecasting method for intermittent demand; forecasts demand size and inter-arrival interval separately. |
| **Holt-Winters** | Triple exponential smoothing — separate smoothed components for level, trend and seasonality. |
| **Rolling-origin CV** | Time-series validation: train to time t, predict forward, roll t, repeat. |

---

## Part 3 — Direct answers to likely questions

### Forecasting

**Q: Why WMAPE instead of MAPE?**
MAPE divides each error by that period's actual. FMCG demand has many
zero-demand days, so MAPE is either undefined (divide by zero) or explodes on
near-zero days — a single unit of error on a day with demand 1 is 100%, and it
drowns out everything else. WMAPE pools all absolute errors and divides once by
total actual demand, which is well-defined with zeros and automatically weights
high-volume SKUs more heavily. That weighting is correct here: being wrong on a
top seller matters more than on a slow mover.

**Q: Why can't you use K-fold cross-validation?**
K-fold shuffles randomly, so a fold could train on data from June and test on
May. The model would see the future while predicting the past — leakage. The
reported accuracy would be optimistic and would collapse in production. I used
rolling-origin: train on everything up to a cutoff, predict the next 28 days,
move the cutoff forward, repeat for 3 folds. That mirrors how a model gets
retrained in production.

**Q: Why report improvement over seasonal naive rather than raw WMAPE?**
A raw WMAPE of 0.70 is meaningless without context — it could be excellent or
terrible depending on how noisy the series is. Seasonal naive (same weekday
last week) is free and requires no model. If a gradient boosting model can't
beat it, the complexity isn't earning anything. Holt-Winters beat it by 24.6%,
which is what justifies deploying a model at all.

**Q: Holt-Winters beat LightGBM. Isn't that a red flag?**
It's a real result and I kept it. Two reasons it happened. First, at SKU-day
granularity each series is short and noisy, and Holt-Winters fits per SKU, so
it captures each SKU's own level and weekly seasonality directly instead of
learning it from features. Second, LightGBM showed a bias of +5.58 units —
systematic under-forecasting. In an inventory setting under-forecasting causes
stockouts directly, so I'd reject it here even if the WMAPE gap were smaller.
With longer history, promotion and price features, and per-segment models, I'd
expect LightGBM to overtake it.

**Q: What is leakage in your feature engineering, and how did you avoid it?**
A rolling mean computed on the current row includes today's demand, which is
the target. I shift by one day *before* rolling, so the window ends yesterday:
`g.shift(1).rolling(7).mean()`, never `g.rolling(7).mean()`. Every lag feature
is at least lag-1 for the same reason.

**Q: Why did you reindex to a complete calendar?**
Days with no orders don't exist as rows in the raw aggregation. If you compute
a lag-7 feature on that gappy frame, "seven rows back" isn't "seven days back"
— it's seven *observed* days back, which silently shifts every feature by a
varying amount. I reindex each SKU to a complete date range and fill zeros, so
absence of demand is recorded as zero demand.

**Q: How would you forecast the C-class intermittent SKUs?**
Croston's method or its bias-corrected variant SBA (Syntetos-Boylan
Approximation). Croston splits the series into demand size when it occurs and
the interval between demand occurrences, smooths each separately, and divides.
Standard exponential smoothing fails on intermittent series because it decays
the forecast toward zero between demand events. I excluded C-class from the
forecasting module rather than report bad numbers on it, which is why it's in
the limitations section.

### Segmentation

**Q: Walk me through ABC classification.**
Sort SKUs by revenue descending. Compute each SKU's revenue share, then the
running cumulative share. A is every SKU up to 80% cumulative, B up to 95%, C
the rest. The crucial detail: the cut is on the *cumulative* share, not the
individual SKU's share. My result was 21% of SKUs producing 79% of revenue — a
textbook Pareto.

**Q: Why weekly CV for XYZ instead of daily?**
Daily FMCG demand is dominated by structural zeros — a SKU that sells in three
of seven days looks wildly variable on a daily basis even if weekly volume is
rock steady. Daily CV would mostly measure order frequency, not planning risk.
Planners replenish on a weekly rhythm, so weekly buckets measure variability at
the horizon that actually matters. With daily buckets I got zero Z-class SKUs;
with weekly I got 39, which matched the visible lumpiness.

**Q: What do you actually do with the 9 cells?**
AX — high value, predictable — gets automated replenishment with a thin buffer;
it's where automation pays. AZ — high value, erratic — gets manual planner
review and a hedged supply agreement, because it carries the most inventory
risk per rupee. CZ — low value, erratic — is make-to-order or delist; buffering
it to 98% ties up capital protecting almost no revenue. That mapping is the
deliverable; the labels alone are just a spreadsheet.

### Inventory

**Q: Derive the safety stock formula.**
Demand over the lead time is a random sum: L days of demand where L is itself
random. Its variance has two components. If lead time were fixed, variance
would be L × sigma_d². If demand were fixed at its mean, variance from
lead-time variability would be d_bar² × sigma_L². Those two sources are
independent, so variances add:

```
Var(demand over lead time) = L_bar * sigma_d^2 + d_bar^2 * sigma_L^2
SS = z * sqrt( that )
```

The square root sits outside because you add variances and then convert back to
a standard deviation. z is the standard normal quantile at your target cycle
service level.

**Q: Why is z = 1.645 for 95%?**
It's the one-sided 95th percentile of the standard normal. You only care about
demand being *higher* than expected — excess stock isn't a service failure — so
it's one-tailed. 90% → 1.282, 95% → 1.645, 98% → 2.054, 99% → 2.326.

**Q: What does the formula assume, and when does it break?**
It assumes demand over lead time is approximately normal. That fails for
intermittent demand, where the distribution is heavily skewed with a spike at
zero, so the normal-based safety stock is wrong — usually over-stocked at high
service targets. For those SKUs I'd fit an empirical or negative-binomial
distribution and take the quantile directly. It also assumes demand and lead
time are independent; if suppliers slow down exactly when everyone orders more,
the true variance is higher than the formula gives.

**Q: Cycle service level vs fill rate — what's the difference?**
Cycle service level is the probability of not stocking out *during a
replenishment cycle*. Fill rate is the fraction of demanded *units* served.
They diverge because cycle service level doesn't care how badly you stocked
out. A SKU that stocks out in 10% of cycles but misses only two units each time
has a poor cycle service level and an excellent fill rate. Fill rate is what
the business feels, which is why I simulated rather than trusting the formula —
my simulated fill rate was 99.17% at segment-differentiated targets averaging
around 94%.

**Q: Derive EOQ.**
Total cost = ordering cost + holding cost = (D/Q)·S + (Q/2)·H, where D is
annual demand, Q the order size, S cost per order, H annual holding cost per
unit. Differentiate with respect to Q, set to zero:
−DS/Q² + H/2 = 0, so Q* = sqrt(2DS/H). Practical point: the total cost curve is
very flat near the optimum, so being 20% off EOQ costs only about 2% in total
cost — which is why nobody agonises over the exact number.

**Q: Why simulate instead of trusting the analytical result?**
Three reasons. The formula gives cycle service level, not fill rate. It assumes
normality that intermittent SKUs violate. And it says nothing about the
interaction between order quantity and the reorder point — a large Q keeps you
above the reorder point longer and effectively raises service beyond what SS
alone implies. The simulation runs actual held-out demand through a
continuous-review (s, Q) policy with stochastic lead times and measures fill
rate directly.

**Q: Why does inventory position drive the reorder decision, not on-hand stock?**
Inventory position is on-hand plus on-order. If you trigger on on-hand alone,
you'd reorder every single day while the first shipment is still in transit,
because on-hand stays low until it arrives. You'd massively over-order. That's
a classic simulation bug.

**Q: Your optimised policy has a *lower* fill rate. Isn't that worse?**
It's a deliberate trade, and it's the point of the analysis. The flat 21-day
buffer achieved 99.92% but at 137K in capital. The segmented policy holds
99.17% at 90K — 34.6% less capital for 0.75 percentage points of service.
Whether that's right depends on the cost of a stockout versus the cost of
capital, which is a business decision, not an analyst's. What I provide is the
service-level-versus-cost curve so someone can choose their point on it
knowingly. The curve is convex — each additional percentage point of service
costs more than the last.

### Statistics

**Q: Why chi-square, and why also report Cramer's V?**
Chi-square tests whether the late/on-time outcome is independent of a
categorical dimension. But with n over 200,000, almost any difference reaches
significance — p-values become uninformative. Cramer's V gives effect size on a
0-to-1 scale, so I can say shipping mode (V = 0.246) matters substantially more
than region (0.186), and category (0.005, p = 0.32) doesn't matter at all. The
p-value tells you an effect exists; V tells you whether to care.

**Q: What was your most useful finding?**
That product category was *not* significant. It's a negative result, and it
redirected the whole analysis — it means you don't fix this by reformulating
packaging or changing which products you stock. You fix it in the logistics
network, specifically Standard Class. Two-thirds of late orders sit in one
shipping mode.

### Data engineering

**Q: How did you handle cancelled orders?**
Excluded them from the OTIF denominator. A cancelled order never physically
shipped, so counting it as a delivery failure would inflate the late rate and
misattribute a commercial cancellation to the logistics team. I kept them in
the raw table and flagged them with `is_fulfilled`, so the exclusion is
explicit and reversible rather than a silent row drop.

**Q: The real Kaggle file is latin-1 encoded. How did you handle it?**
Read it with `encoding="latin-1"` explicitly. Reading it as UTF-8 raises a
`UnicodeDecodeError` on the non-ASCII characters in the customer and product
name fields. I also mapped its column names onto a snake_case schema in one
dictionary, so every downstream module is source-agnostic and the synthetic
fallback and real file are interchangeable.

---

## Part 4 — Code you should be able to explain line by line

**ABC classification**
```python
rev = df.groupby('sku')['sales'].sum().sort_values(ascending=False)
cum = rev.cumsum() / rev.sum()
abc = pd.cut(cum, bins=[0, 0.80, 0.95, 1.0], labels=['A','B','C'])
```
`cumsum()/sum()` gives each SKU's running share of total revenue *once sorted
descending* — the sort is load-bearing, without it the cumulative share is
meaningless. `pd.cut` then slices that running share at 80% and 95%.

**Leak-free rolling feature**
```python
d['roll_mean_7'] = g.shift(1).rolling(7).mean()
```
`shift(1)` first, *then* roll. The window ends yesterday. Without the shift the
window includes today's demand, which is the target being predicted.

**Safety stock**
```python
var = L_mean * sigma_d**2 + d_bar**2 * L_std**2
ss = z * np.sqrt(var)
```
Two independent variance terms added, then a single square root, then scaled by
the service-level z-score.

**SQL running total for ABC**
```sql
SUM(revenue) OVER (ORDER BY revenue DESC
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
  / SUM(revenue) OVER ()
```
The framed window gives the running total; the empty `OVER ()` gives the grand
total, in one pass with no self-join. The classification must go in a separate
CTE because window functions are evaluated after the select list is built, so
you can't reference one inside a CASE in the same SELECT.

---

## Part 5 — Questions to ask them

- How do you currently set safety stock — days of cover, or a statistical model?
- Do you plan on cycle service level or fill rate, and is that consistent across categories?
- How are forecasts measured today, and against what baseline?
- Is the demand planning team measured on forecast accuracy or on service outcomes?

---

## Part 6 — Honesty guardrails

If asked whether the data is real: say plainly that the pipeline is built
against the DataCo Smart Supply Chain schema and runs on the real Kaggle file,
and that the committed figures came from a synthetic generator included in the
repo so it runs without the download. Then offer the real-data numbers.

Never claim a business impact you didn't measure. The 34.6% capital reduction
is a *simulated* result under stated assumptions (25% holding rate,
7-day ±2 lead time, lost sales not backorders). Say "simulated" out loud. An
interviewer who catches you overstating that will discount everything else.
