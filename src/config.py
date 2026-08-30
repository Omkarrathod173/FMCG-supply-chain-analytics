"""Central configuration: paths, business parameters, model settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"

for _p in (DATA_RAW, DATA_PROCESSED, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data source
# Put the real Kaggle DataCo file here as data/raw/DataCoSupplyChainDataset.csv
# If it is absent, the pipeline falls back to the synthetic generator so the
# repo runs end-to-end out of the box.
REAL_DATA_FILE = DATA_RAW / "DataCoSupplyChainDataset.csv"
SYNTH_DATA_FILE = DATA_RAW / "synthetic_supply_chain.csv"

RANDOM_SEED = 42

# ------------------------------------------------------------ ABC-XYZ cutoffs
ABC_CUTOFFS = (0.80, 0.95)      # cumulative revenue share: A <=80%, B <=95%, C rest
XYZ_CUTOFFS = (0.50, 1.00)      # coefficient of variation: X <0.5, Y <1.0, Z >=1.0

# ------------------------------------------------------------------ inventory
SERVICE_LEVELS = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
TARGET_SERVICE_LEVEL = 0.95
UNIT_HOLDING_COST_RATE = 0.25   # annual holding cost as a fraction of unit value
LEAD_TIME_MEAN_DAYS = 7.0
LEAD_TIME_STD_DAYS = 2.0

# ---------------------------------------------------------------- forecasting
FORECAST_HORIZON = 28           # days held out in each rolling-origin fold
N_CV_FOLDS = 3
SEASONAL_PERIOD = 7             # weekly seasonality on daily data
