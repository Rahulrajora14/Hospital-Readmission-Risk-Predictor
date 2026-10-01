# 📂 Data folder

| Folder | What goes here | In Git? |
|---|---|---|
| `raw/` | `diabetic_data.csv`, downloaded by `python main.py --step download` | ❌ no (19 MB, re-downloadable) |
| `processed/` | `clean_data.csv`, created by the clean step | ❌ no (re-created) |
| `sample/` | `test_patients.csv`, 300 held-out test patients used by the demo app | ✅ yes |
| `readmission.db` | SQLite database created by the sql step | ❌ no (re-created) |

**Source:** [Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296), UCI Machine Learning Repository (CC BY 4.0).
Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates.* BioMed Research International.
