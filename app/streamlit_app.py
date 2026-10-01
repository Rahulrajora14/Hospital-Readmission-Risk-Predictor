"""
Streamlit dashboard - Hospital Readmission Risk Check.

Run from the project root:
    streamlit run app/streamlit_app.py

Tabs:
  1. Check a patient   - enter details (or load a real test patient),
                         get a risk score + SHAP explanation
  2. Model performance - comparison table and evaluation charts
  3. About             - what the project is and its limits
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import streamlit as st
import xgboost

# make "src" importable when Streamlit runs this file directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.explain.shap_explainer import ReadmissionExplainer, plot_patient_explanation  # noqa: E402
from src.features.engineer import ALL_FEATURES  # noqa: E402

st.set_page_config(page_title="Readmission Risk Check", page_icon="🏥", layout="wide")


# ---------------------------------------------------------------------------
# Load artefacts once (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not config.MODEL_PATH.exists():
        return None, None, None, None
    pipeline = joblib.load(config.MODEL_PATH)
    meta = json.loads(config.METADATA_PATH.read_text())
    explainer = ReadmissionExplainer(pipeline)
    sample = pd.read_csv(config.SAMPLE_PATIENTS_CSV) if config.SAMPLE_PATIENTS_CSV.exists() else None
    return pipeline, meta, explainer, sample


pipeline, meta, explainer, sample_patients = load_artifacts()

if pipeline is None:
    st.error("No trained model found. Run `python main.py` from the project root first, "
             "then reload this page.")
    st.stop()

# A model pickled with other library versions can load fine but predict wrong numbers,
# so compare major.minor versions and stop with clear instructions if they differ.
def _major_minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


saved = meta.get("library_versions", {})
installed = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}
mismatched = [f"{lib}: model saved with {saved[lib]}, you have {installed[lib]}"
              for lib in installed
              if lib in saved and _major_minor(saved[lib]) != _major_minor(installed[lib])]
if mismatched:
    st.error("The saved model was trained with different library versions, so its "
             "predictions would not be reliable on your setup.\n\n"
             + "\n".join(f"- {m}" for m in mismatched)
             + "\n\nRetrain it on your machine (about 3 minutes), then reload this page:\n\n"
             "`python main.py`")
    st.stop()

THRESHOLD = meta["threshold"]
OPTIONS = meta["category_options"]
RANGES = meta["numeric_ranges"]
AGE_LABELS = {a: f"{a - 5}-{a + 5}" for a in range(5, 100, 10)}

# Widget keys = feature names, so a sample patient can fill the form directly
NUMERIC_INPUTS = [f for f in meta["numeric_features"] if f != "total_prior_visits"]


def set_defaults():
    for f in NUMERIC_INPUTS:
        st.session_state.setdefault(f, RANGES[f][2])
    for f, v in meta["category_defaults"].items():
        st.session_state.setdefault(f, v)
    st.session_state.setdefault("actual_outcome", None)


def load_random_patient():
    row = sample_patients.sample(1).iloc[0]
    for f in NUMERIC_INPUTS:
        st.session_state[f] = int(row[f])
    for f in meta["categorical_features"]:
        st.session_state[f] = str(row[f])
    st.session_state["actual_outcome"] = int(row[config.TARGET])


def reset_form():
    for f in NUMERIC_INPUTS + meta["categorical_features"] + ["actual_outcome"]:
        st.session_state.pop(f, None)


set_defaults()


def num_input(label, feature, col):
    lo, hi, _ = RANGES[feature]
    col.number_input(label, min_value=lo, max_value=hi, step=1, key=feature)


def cat_input(label, feature, col):
    col.selectbox(label, OPTIONS[feature], key=feature)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏥 Readmission risk check")
st.caption("Estimates how likely a diabetic patient is to be readmitted within 30 days of "
           f"discharge, and explains why. Model: {meta['model_name']}, trained on ~49k "
           "real US hospital encounters.")

tab_check, tab_perf, tab_about = st.tabs(["Check a patient", "Model performance", "About"])

# ---------------------------------------------------------------------------
# Tab 1 - single patient
# ---------------------------------------------------------------------------
with tab_check:
    b1, b2, _ = st.columns([1.3, 0.8, 4])
    b1.button("🎲 Load a real test patient", on_click=load_random_patient,
              disabled=sample_patients is None)
    b2.button("Reset form", on_click=reset_form)

    # The result is drawn here (above the form) but filled in after the form
    result_area = st.container()
    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.subheader("Patient and stay")
    c1.selectbox("Age group", list(AGE_LABELS), format_func=AGE_LABELS.get, key="age_years")
    cat_input("Gender", "gender", c1)
    num_input("Days in hospital", "time_in_hospital", c1)
    cat_input("Admission type", "admission_type", c1)
    cat_input("Admission source", "admission_source_group", c1)
    cat_input("Discharge destination", "discharge_group", c1)
    cat_input("Admitting specialty", "medical_specialty", c1)

    c2.subheader("Clinical picture")
    cat_input("Primary diagnosis", "diag_1_group", c2)
    cat_input("Secondary diagnosis", "diag_2_group", c2)
    cat_input("Additional diagnosis", "diag_3_group", c2)
    num_input("Number of diagnoses", "number_diagnoses", c2)
    num_input("Lab procedures", "num_lab_procedures", c2)
    num_input("Procedures", "num_procedures", c2)
    cat_input("HbA1c test", "A1Cresult", c2)
    cat_input("Glucose test", "max_glu_serum", c2)

    c3.subheader("History and medication")
    num_input("Inpatient visits, last year", "number_inpatient", c3)
    num_input("Emergency visits, last year", "number_emergency", c3)
    num_input("Outpatient visits, last year", "number_outpatient", c3)
    num_input("Medications given", "num_medications", c3)
    num_input("Diabetes meds active", "num_meds_active", c3)
    num_input("Diabetes meds changed", "num_meds_changed", c3)
    cat_input("Insulin", "insulin", c3)
    cat_input("Metformin", "metformin", c3)
    cat_input("Any medication changed", "change", c3)
    cat_input("On diabetes medication", "diabetesMed", c3)

    # Build the one-row DataFrame the model expects
    patient = {f: st.session_state[f] for f in NUMERIC_INPUTS + meta["categorical_features"]}
    patient["total_prior_visits"] = (patient["number_inpatient"] + patient["number_emergency"]
                                     + patient["number_outpatient"])
    patient_df = pd.DataFrame([patient])[ALL_FEATURES]

    score = float(pipeline.predict_proba(patient_df)[:, 1][0])
    flagged = score >= THRESHOLD
    avg = meta["positive_rate"]

    r1, r2 = result_area.columns([1, 2.2])
    with r1:
        st.subheader("Result")
        st.metric("Risk score", f"{score:.1%}",
                  delta=f"{(score - avg) * 100:+.1f} pts vs average patient ({avg:.1%})",
                  delta_color="inverse")
        if flagged:
            st.error(f"**Flag for follow-up.** Score is above the model's alert threshold "
                     f"of {THRESHOLD:.1%}.")
        else:
            st.success(f"**Below the follow-up threshold** of {THRESHOLD:.1%}.")
        if st.session_state.get("actual_outcome") is not None:
            outcome = "was readmitted within 30 days" if st.session_state["actual_outcome"] \
                else "was not readmitted within 30 days"
            st.info(f"This is a real held-out test patient. In reality the patient {outcome}.")
        st.caption("The alert threshold was tuned on validation data to balance catching "
                   "readmissions against false alarms.")

    with r2:
        table = explainer.explain_patient(patient_df, top_n=8)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        plot_patient_explanation(table, score, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with result_area.expander("See the factor table"):
        show = table.copy()
        show["value"] = show["value"].astype(str)
        show["impact"] = show["impact"].round(3)
        st.dataframe(show, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Tab 2 - performance
# ---------------------------------------------------------------------------
with tab_perf:
    st.subheader("Model comparison on the held-out test set")
    if config.COMPARISON_PATH.exists():
        comp = pd.read_csv(config.COMPARISON_PATH)
        st.dataframe(comp, hide_index=True, width="stretch")
    m = meta["test_metrics"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    k2.metric("PR-AUC", f"{m['pr_auc']:.3f}", help=f"Random guessing scores about {avg:.3f}")
    k3.metric("Recall at threshold", f"{m['recall']:.1%}")
    k4.metric("Top-10% lift", f"{m.get('top10_lift', 0):.2f}x",
              help="Readmission rate among the 10% highest-scored patients, divided by the "
                   "average rate")

    fig_dir = config.FIGURES_DIR
    g1, g2 = st.columns(2)
    for col, name, cap in [
        (g1, "roc_curves.png", "ROC curves"),
        (g2, "pr_curves.png", "Precision-recall curves"),
        (g1, "shap_global_importance.png", "Top risk drivers (SHAP)"),
        (g2, "confusion_matrix.png", "Confusion matrix of the selected model"),
    ]:
        if (fig_dir / name).exists():
            col.image(str(fig_dir / name), caption=cap, width="stretch")

# ---------------------------------------------------------------------------
# Tab 3 - about
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
**What this is.** A machine-learning project that predicts 30-day hospital readmission for
diabetic patients, using the public *Diabetes 130-US Hospitals (1999–2008)* dataset from the
UCI Machine Learning Repository.

**Why it matters.** Readmissions within 30 days are expensive for payers and often avoidable
with a follow-up call, medication review, or home visit. A ranked risk list helps a care team
spend limited time on the patients most likely to come back.

**How it was built.** SQL exploration in SQLite, leakage-aware cleaning (one encounter per
patient, deceased and hospice patients removed), clinical feature engineering (ICD-9 grouping,
medication-change counts), three models compared (Logistic Regression, Random Forest,
Optuna-tuned XGBoost), threshold tuned on validation data, and SHAP explanations.

**Limits.** Educational project on 1999–2008 data. Not validated for clinical use. Race is
intentionally excluded from the features.
        """
    )
