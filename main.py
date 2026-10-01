"""
One command to run the whole pipeline:

    python main.py                 # run every step
    python main.py --step train    # run a single step
    python main.py --from clean    # run from a step to the end

Steps, in order:
    download -> sql -> eda -> clean -> train -> explain
"""
import argparse
import time

from src.data.download_data import download_dataset
from src.data.load_to_sql import load_csv_to_sqlite
from src.data.sql_eda import run_sql_eda
from src.explain.shap_explainer import run_explanations
from src.features.build_dataset import build_dataset
from src.models.train import train_and_compare

STEPS = {
    "download": ("Download the raw dataset", download_dataset),
    "sql": ("Load CSV into SQLite", load_csv_to_sqlite),
    "eda": ("Exploratory analysis with SQL", run_sql_eda),
    "clean": ("Clean data + engineer features", build_dataset),
    "train": ("Train, tune and compare models", train_and_compare),
    "explain": ("Explain the model with SHAP", run_explanations),
}


def run(selected: list[str]) -> None:
    total_start = time.time()
    for i, name in enumerate(selected, 1):
        title, func = STEPS[name]
        print(f"\n{'=' * 64}\n[{i}/{len(selected)}] {title}\n{'=' * 64}")
        start = time.time()
        func()
        print(f"  done in {time.time() - start:.1f}s")
    print(f"\nPipeline finished in {time.time() - total_start:.1f}s")
    print("Next: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hospital readmission pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step", choices=STEPS, help="run only this step")
    group.add_argument("--from", dest="start", choices=STEPS, help="run from this step onward")
    args = parser.parse_args()

    names = list(STEPS)
    if args.step:
        names = [args.step]
    elif args.start:
        names = names[names.index(args.start):]
    run(names)
