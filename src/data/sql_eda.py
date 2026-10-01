"""
Step 3 - Exploratory Data Analysis using SQL.

Reads sql/eda_queries.sql, runs every named query against SQLite,
prints the result, saves each one as a CSV in reports/sql_results/,
and draws a few charts into reports/figures/.
"""
import re
import sqlite3

import matplotlib

matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from src import config


def parse_sql_file(path=config.EDA_SQL_FILE) -> dict[str, str]:
    """Split the .sql file into {query_name: query_text}."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^--\s*name:\s*(\w+)", text, flags=re.MULTILINE)
    # re.split gives [preamble, name1, body1, name2, body2, ...]
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_queries() -> dict[str, pd.DataFrame]:
    queries = parse_sql_file()
    config.SQL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    with sqlite3.connect(config.DB_PATH) as conn:
        for name, query in queries.items():
            df = pd.read_sql_query(query, conn)
            results[name] = df
            df.to_csv(config.SQL_RESULTS_DIR / f"{name}.csv", index=False)
            print(f"\n--- {name} ---")
            print(df.to_string(index=False))

    return results


def _bar_chart(df, x, y, title, xlabel, filename, color="#2a7f8f"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df[x].astype(str), df[y], color=color)
    ax.set_title(title, fontsize=13, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("30-day readmission rate (%)")
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(df[y]):
        ax.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130)
    plt.close(fig)


def make_charts(results: dict[str, pd.DataFrame]) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    _bar_chart(results["03_readmission_by_age"], "age_group", "readmit_30d_rate_pct",
               "Readmission rate by age group", "Age group", "eda_readmission_by_age.png")

    _bar_chart(results["04_readmission_by_prior_inpatient_visits"], "prior_inpatient_visits",
               "readmit_30d_rate_pct", "Readmission rate by prior inpatient visits",
               "Inpatient visits in the previous year", "eda_readmission_by_prior_visits.png",
               color="#c0504d")

    _bar_chart(results["06_readmission_by_length_of_stay"], "days_in_hospital",
               "readmit_30d_rate_pct", "Readmission rate by length of stay",
               "Days in hospital", "eda_readmission_by_length_of_stay.png", color="#4f6d8f")

    print(f"\n  Charts saved -> {config.FIGURES_DIR}")


def run_sql_eda() -> None:
    results = run_queries()
    make_charts(results)


if __name__ == "__main__":
    run_sql_eda()
