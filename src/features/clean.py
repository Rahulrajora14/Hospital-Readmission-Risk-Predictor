"""
Step 4a - Data cleaning.

Rules applied (in this order), each one explained:

1. '?' -> NaN
   The dataset uses '?' for missing values. We convert it to a real NaN.

2. Remove patients who died or went to hospice
   They cannot be readmitted, so keeping them would teach the model a
   false pattern ("these patients never come back").

3. Keep only the FIRST encounter of each patient
   Many patients appear several times. If visit #1 lands in the training
   set and visit #2 in the test set, the model has already "seen" that
   patient -> test score looks better than reality (data leakage).

4. Drop rows with invalid gender (only 1-3 rows).

5. Create the target: 1 if readmitted in under 30 days, else 0.
"""
import numpy as np
import pandas as pd

from src import config

# discharge_disposition_id values meaning expired (11, 19, 20, 21) or hospice (13, 14)
EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    log = [("Raw encounters", len(df))]

    # 1. '?' is the dataset's missing-value marker
    df = df.replace("?", np.nan)

    # 2. Patients who expired or went to hospice cannot be readmitted
    df["discharge_disposition_id"] = df["discharge_disposition_id"].astype(int)
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)]
    log.append(("After removing expired / hospice", len(df)))

    # 3. One row per patient (their first visit) -> no patient leakage
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    log.append(("After keeping first visit per patient", len(df)))

    # 4. Invalid gender values
    df = df[df["gender"].isin(["Male", "Female"])]
    log.append(("After removing invalid gender", len(df)))

    # 5. Binary target
    df[config.TARGET] = (df["readmitted"] == "<30").astype(int)

    if verbose:
        for step, n in log:
            print(f"  {step:<42} {n:>8,}")
        print(f"  {'Positive rate (readmitted < 30 days)':<42} {df[config.TARGET].mean():>8.2%}")

    return df.reset_index(drop=True)
