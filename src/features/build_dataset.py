"""
Step 4 - Build the model-ready dataset.

SQLite (raw table)  ->  clean_data()  ->  engineer_features()  ->  clean_data.csv
"""
from src import config
from src.data.load_to_sql import read_table
from src.features.clean import clean_data
from src.features.engineer import engineer_features


def build_dataset():
    print("  Reading raw table from SQLite...")
    raw = read_table()

    print("  Cleaning:")
    cleaned = clean_data(raw)

    print("  Engineering features...")
    features = engineer_features(cleaned)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(config.CLEAN_CSV, index=False)
    print(f"  Final dataset: {features.shape[0]:,} rows x {features.shape[1]} columns "
          f"-> {config.CLEAN_CSV}")
    return features


if __name__ == "__main__":
    build_dataset()
