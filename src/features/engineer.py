"""
Step 4b - Feature engineering.

Turns raw hospital columns into model-ready features. Every feature
here has a clinical / business reason behind it (see docs/HOW_IT_WORKS.md).
"""
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Final feature lists - the model sees exactly these columns
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "age_years",              # age bracket converted to a number
    "time_in_hospital",       # length of stay (days)
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",      # visits in the previous year
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "total_prior_visits",     # engineered: outpatient + emergency + inpatient
    "num_meds_changed",       # engineered: diabetes meds dosed up or down
    "num_meds_active",        # engineered: diabetes meds the patient is on
]

CATEGORICAL_FEATURES = [
    # NOTE: race is deliberately NOT used as a feature (fairness - see docs)
    "gender",
    "admission_type",         # Emergency / Urgent / Elective ...
    "discharge_group",        # Home / Home health / Facility ...
    "admission_source_group", # Referral / Emergency room / Transfer ...
    "medical_specialty",      # top 10 kept, rest grouped as Other
    "diag_1_group",           # primary diagnosis, ICD-9 grouped
    "diag_2_group",
    "diag_3_group",
    "max_glu_serum",          # glucose test result
    "A1Cresult",              # HbA1c test result
    "insulin",                # No / Steady / Up / Down
    "metformin",
    "change",                 # any diabetes medication changed?
    "diabetesMed",            # on any diabetes medication?
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# The 23 diabetes medication columns in the raw data
MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# ---------------------------------------------------------------------------
# Lookup tables (ID -> readable name), from the dataset's IDs_mapping.csv
# ---------------------------------------------------------------------------
AGE_TO_YEARS = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}

ADMISSION_TYPE = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
    5: "Unknown", 6: "Unknown", 7: "Trauma Center", 8: "Unknown",
}

DISCHARGE_GROUP = {
    1: "Home",
    6: "Home with health service", 8: "Home with health service",
    2: "Transferred to facility", 3: "Transferred to facility",
    4: "Transferred to facility", 5: "Transferred to facility",
    10: "Transferred to facility", 15: "Transferred to facility",
    22: "Transferred to facility", 23: "Transferred to facility",
    24: "Transferred to facility", 27: "Transferred to facility",
    28: "Transferred to facility", 29: "Transferred to facility",
    30: "Transferred to facility",
    7: "Left against medical advice",
}

ADMISSION_SOURCE_GROUP = {
    1: "Referral", 2: "Referral", 3: "Referral",
    7: "Emergency room",
    4: "Transfer", 5: "Transfer", 6: "Transfer", 10: "Transfer",
    18: "Transfer", 22: "Transfer", 25: "Transfer", 26: "Transfer",
}

TOP_N_SPECIALTIES = 10


def map_icd9_to_group(code) -> str:
    """
    Group a raw ICD-9 diagnosis code into one of 9 disease families.
    There are 700+ distinct codes - far too many for a model - so we use
    the standard grouping from Strack et al. (2014), the paper that
    published this dataset.
    """
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):          # supplementary / external causes
        return "Other"
    try:
        value = float(code)
    except ValueError:
        return "Other"

    whole = int(value)
    if whole == 250:
        return "Diabetes"
    if 390 <= whole <= 459 or whole == 785:
        return "Circulatory"
    if 460 <= whole <= 519 or whole == 786:
        return "Respiratory"
    if 520 <= whole <= 579 or whole == 787:
        return "Digestive"
    if 800 <= whole <= 999:
        return "Injury"
    if 710 <= whole <= 739:
        return "Musculoskeletal"
    if 580 <= whole <= 629 or whole == 788:
        return "Genitourinary"
    if 140 <= whole <= 239:
        return "Neoplasms"
    return "Other"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Age: "[70-80)" -> 75 (keeps the natural order of ages) -----------
    df["age_years"] = df["age"].map(AGE_TO_YEARS)

    # --- ID columns -> readable groups -------------------------------------
    df["admission_type"] = df["admission_type_id"].astype(int).map(ADMISSION_TYPE).fillna("Unknown")
    df["discharge_group"] = df["discharge_disposition_id"].astype(int).map(DISCHARGE_GROUP).fillna("Other")
    df["admission_source_group"] = df["admission_source_id"].astype(int).map(ADMISSION_SOURCE_GROUP).fillna("Other")

    # --- Diagnoses: 700+ ICD-9 codes -> 9 disease groups -------------------
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_group"] = df[col].apply(map_icd9_to_group)

    # --- Medical specialty: keep top 10, group the long tail ---------------
    top = df["medical_specialty"].value_counts().nlargest(TOP_N_SPECIALTIES).index
    df["medical_specialty"] = df["medical_specialty"].where(
        df["medical_specialty"].isin(top) | df["medical_specialty"].isna(), "Other"
    ).fillna("Missing")

    # --- Lab tests: 'None' means the test was not done ----------------------
    for col in ["max_glu_serum", "A1Cresult"]:
        df[col] = df[col].fillna("None").replace("None", "Not measured")

    # --- 'Ch' / 'No' -> readable 'Yes' / 'No' ------------------------------
    df["change"] = df["change"].replace({"Ch": "Yes"})

    # --- Medication summary features ----------------------------------------
    meds = [c for c in MEDICATION_COLUMNS if c in df.columns]
    df["num_meds_changed"] = df[meds].isin(["Up", "Down"]).sum(axis=1)
    df["num_meds_active"] = (df[meds] != "No").sum(axis=1)

    # --- Healthcare utilisation in the previous year ------------------------
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    # Make sure numeric columns really are numeric
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].apply(pd.to_numeric)

    return df[ALL_FEATURES + [config.TARGET]]
