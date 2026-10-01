"""
Step 2 - Load the raw CSV into a SQLite database.

Why a database at all? In real companies data lives in SQL databases
(or Databricks / Snowflake tables), not in CSV files. So from this point on
the pipeline reads data with SQL, just like it would at work.

SQLite needs no installation - the whole database is one file:
data/readmission.db
"""
import sqlite3

import pandas as pd

from src import config

TABLE_NAME = "encounters"


def load_csv_to_sqlite() -> None:
    if not config.RAW_CSV.exists():
        raise FileNotFoundError(
            f"{config.RAW_CSV} not found. Run the download step first."
        )

    # keep_default_na=False keeps the dataset's own markers ('?' and 'None')
    # as plain text, so the database holds exactly what the source file holds.
    df = pd.read_csv(config.RAW_CSV, keep_default_na=False, low_memory=False)

    with sqlite3.connect(config.DB_PATH) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        # Indexes make the GROUP BY / JOIN queries in the EDA step fast
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_patient ON {TABLE_NAME}(patient_nbr)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_readmit ON {TABLE_NAME}(readmitted)")
        n_rows = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]

    print(f"  Loaded {n_rows:,} rows into table '{TABLE_NAME}' -> {config.DB_PATH}")


def read_table(query: str = f"SELECT * FROM {TABLE_NAME}") -> pd.DataFrame:
    """Run any SQL query against the database and get a DataFrame back."""
    with sqlite3.connect(config.DB_PATH) as conn:
        return pd.read_sql_query(query, conn)


if __name__ == "__main__":
    load_csv_to_sqlite()
