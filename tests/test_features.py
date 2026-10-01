"""
Unit tests for cleaning and feature engineering.
Run with:  pytest -v
They use tiny hand-made DataFrames, so no dataset download is needed.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.features.clean import clean_data
from src.features.engineer import ALL_FEATURES, map_icd9_to_group, engineer_features, MEDICATION_COLUMNS


@pytest.mark.parametrize("code, expected", [
    ("250.83", "Diabetes"),
    ("428", "Circulatory"),
    ("785", "Circulatory"),
    ("486", "Respiratory"),
    ("786.5", "Respiratory"),
    ("560", "Digestive"),
    ("820", "Injury"),
    ("715", "Musculoskeletal"),
    ("599", "Genitourinary"),
    ("162", "Neoplasms"),
    ("V45", "Other"),
    ("E888", "Other"),
    (np.nan, "Missing"),
])
def test_icd9_grouping(code, expected):
    assert map_icd9_to_group(code) == expected


def _raw_rows():
    base = {
        "encounter_id": 1, "patient_nbr": 100, "race": "Caucasian", "gender": "Female",
        "age": "[70-80)", "weight": "?", "admission_type_id": 1, "discharge_disposition_id": 1,
        "admission_source_id": 7, "time_in_hospital": 3, "payer_code": "?",
        "medical_specialty": "Cardiology", "num_lab_procedures": 40, "num_procedures": 1,
        "num_medications": 15, "number_outpatient": 0, "number_emergency": 1,
        "number_inpatient": 2, "diag_1": "428", "diag_2": "250.01", "diag_3": "V45",
        "number_diagnoses": 9, "max_glu_serum": "None", "A1Cresult": ">8",
        "change": "Ch", "diabetesMed": "Yes", "readmitted": "<30",
    }
    base.update({m: "No" for m in MEDICATION_COLUMNS})
    base["insulin"] = "Up"
    rows = [dict(base)]
    rows.append({**base, "encounter_id": 2, "readmitted": "NO"})               # same patient again
    rows.append({**base, "encounter_id": 3, "patient_nbr": 200,
                 "discharge_disposition_id": 11})                              # expired
    rows.append({**base, "encounter_id": 4, "patient_nbr": 300,
                 "gender": "Unknown/Invalid"})                                 # invalid gender
    rows.append({**base, "encounter_id": 5, "patient_nbr": 400, "readmitted": ">30"})
    return pd.DataFrame(rows)


def test_clean_removes_leakage_and_invalid_rows():
    out = clean_data(_raw_rows(), verbose=False)
    assert out["patient_nbr"].tolist() == [100, 400]     # first visit kept, others dropped
    assert out[config.TARGET].tolist() == [1, 0]         # '>30' is NOT a positive


def test_engineered_features():
    out = engineer_features(clean_data(_raw_rows(), verbose=False))
    row = out.iloc[0]
    assert list(out.columns) == ALL_FEATURES + [config.TARGET]
    assert row["age_years"] == 75
    assert row["total_prior_visits"] == 3
    assert row["num_meds_changed"] == 1 and row["num_meds_active"] == 1
    assert row["diag_1_group"] == "Circulatory" and row["diag_2_group"] == "Diabetes"
    assert row["max_glu_serum"] == "Not measured"
    assert row["change"] == "Yes"
    assert row["admission_source_group"] == "Emergency room"
    assert out[ALL_FEATURES].isna().sum().sum() == 0
