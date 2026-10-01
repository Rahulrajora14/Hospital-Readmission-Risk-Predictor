"""
Central configuration for the whole project.

Every path and setting lives here, so no other file hard-codes a path.
If you want to change something (number of tuning trials, test size...),
change it here once and the whole pipeline picks it up.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, so it works on any machine)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"

RAW_CSV = RAW_DIR / "diabetic_data.csv"            # original UCI file
DB_PATH = DATA_DIR / "readmission.db"              # SQLite database
CLEAN_CSV = PROCESSED_DIR / "clean_data.csv"       # after cleaning + features
SAMPLE_PATIENTS_CSV = SAMPLE_DIR / "test_patients.csv"  # used by the app

SQL_DIR = ROOT / "sql"
EDA_SQL_FILE = SQL_DIR / "eda_queries.sql"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "readmission_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SQL_RESULTS_DIR = REPORTS_DIR / "sql_results"
METRICS_PATH = REPORTS_DIR / "metrics.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

MLFLOW_DB = ROOT / "mlflow.db"   # MLflow experiment-tracking database

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
# Official UCI download (zip that contains diabetic_data.csv)
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
# Public GitHub mirror of the same CSV, used only if UCI is unreachable
MIRROR_CSV_URL = (
    "https://raw.githubusercontent.com/AkankshaUtreja/"
    "Diabetic-Patients-Readmission-Prediction/master/diabetic_data.csv"
)

# ---------------------------------------------------------------------------
# Modelling settings
# ---------------------------------------------------------------------------
TARGET = "readmitted_30d"     # 1 = readmitted within 30 days, 0 = not
RANDOM_STATE = 42             # fixed seed -> same results every run
TEST_SIZE = 0.15              # 15% of patients held out for the final test
VAL_SIZE = 0.15               # 15% used for tuning + threshold selection
N_TUNING_TRIALS = 30          # Optuna trials for XGBoost
USE_MLFLOW = True             # log experiments if mlflow is installed
