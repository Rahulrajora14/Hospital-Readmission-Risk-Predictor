"""
Step 6 - Explain the model with SHAP.

A healthcare model that just says "risk = 0.31" is not trusted by doctors.
SHAP (SHapley Additive exPlanations) splits every single prediction into
contributions from each feature:

    prediction = base value + contribution(age) + contribution(prior visits) + ...

- Positive contribution  -> pushes risk UP
- Negative contribution  -> pushes risk DOWN

We produce:
1. Global importance  : which features matter most across all patients
2. Beeswarm plot      : how high / low values of a feature move the risk
3. Single patient     : why THIS patient got THIS score (used by the app)
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config
from src.features.engineer import ALL_FEATURES, CATEGORICAL_FEATURES

READABLE_NAMES = {
    "age_years": "Age",
    "time_in_hospital": "Days in hospital",
    "num_lab_procedures": "Lab procedures",
    "num_procedures": "Procedures",
    "num_medications": "Medications given",
    "number_outpatient": "Outpatient visits (prev. year)",
    "number_emergency": "Emergency visits (prev. year)",
    "number_inpatient": "Inpatient visits (prev. year)",
    "number_diagnoses": "Number of diagnoses",
    "total_prior_visits": "Total prior visits",
    "num_meds_changed": "Diabetes meds changed",
    "num_meds_active": "Diabetes meds active",
    "gender": "Gender",
    "admission_type": "Admission type",
    "discharge_group": "Discharge destination",
    "admission_source_group": "Admission source",
    "medical_specialty": "Admitting specialty",
    "diag_1_group": "Primary diagnosis",
    "diag_2_group": "Secondary diagnosis",
    "diag_3_group": "Additional diagnosis",
    "max_glu_serum": "Glucose test",
    "A1Cresult": "HbA1c test",
    "insulin": "Insulin",
    "metformin": "Metformin",
    "change": "Medication changed",
    "diabetesMed": "On diabetes medication",
}


class ReadmissionExplainer:
    """Wraps the trained pipeline + a SHAP TreeExplainer."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.explainer = shap.TreeExplainer(self.model)
        self.encoded_names = list(self.preprocessor.get_feature_names_out())
        self.encoded_to_original = [self._original_feature(n) for n in self.encoded_names]

    @staticmethod
    def _original_feature(encoded_name: str) -> str:
        """'cat__diag_1_group_Circulatory' -> 'diag_1_group'."""
        prefix, rest = encoded_name.split("__", 1)
        if prefix == "num":
            return rest
        # longest matching categorical name wins (avoids partial-name clashes)
        matches = [c for c in CATEGORICAL_FEATURES if rest.startswith(c + "_")]
        return max(matches, key=len)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values per ENCODED column (one-hot columns separate)."""
        X_enc = self.preprocessor.transform(X[ALL_FEATURES])
        return self.explainer.shap_values(X_enc)

    def grouped_shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encoding splits 'Primary diagnosis' into 9 columns.
        We add their SHAP values back together so each ORIGINAL feature
        gets one number (SHAP values are additive, so this is valid).
        """
        values = pd.DataFrame(self.shap_values(X), columns=self.encoded_names)
        return values.T.groupby(self.encoded_to_original, sort=False).sum().T[ALL_FEATURES]

    def explain_patient(self, patient: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
        """Top factors for ONE patient, sorted by impact."""
        contrib = self.grouped_shap_values(patient).iloc[0]
        row = patient.iloc[0]
        table = pd.DataFrame({
            "feature": [READABLE_NAMES.get(f, f) for f in contrib.index],
            "value": [row[f] for f in contrib.index],
            "impact": contrib.values,
        })
        table["direction"] = np.where(table["impact"] > 0, "raises risk", "lowers risk")
        return table.reindex(table["impact"].abs().sort_values(ascending=False).index).head(top_n)

    @property
    def base_value(self) -> float:
        return float(np.ravel(self.explainer.expected_value)[0])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_global_importance(grouped: pd.DataFrame, top_n=15, filename="shap_global_importance.png"):
    importance = grouped.abs().mean().sort_values().tail(top_n)
    labels = [READABLE_NAMES.get(f, f) for f in importance.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, importance.values, color="#2a7f8f")
    ax.set_title("What drives readmission risk (mean |SHAP value|)", loc="left", fontsize=12)
    ax.set_xlabel("Average impact on model output (log-odds)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130)
    plt.close(fig)


def plot_patient_explanation(table: pd.DataFrame, score: float, filename=None, ax=None):
    """Horizontal bar chart of one patient's top factors (red = raises risk)."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    t = table.iloc[::-1]
    labels = [f"{f} = {v}" for f, v in zip(t["feature"], t["value"])]
    colors = ["#c0504d" if v > 0 else "#2a7f8f" for v in t["impact"]]
    ax.barh(labels, t["impact"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Contribution to risk (log-odds)  |  red raises risk, teal lowers it")
    ax.set_title(f"Why this patient scored {score:.1%}", loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    if own_fig:
        fig.tight_layout()
        if filename:
            fig.savefig(config.FIGURES_DIR / filename, dpi=130)
        plt.close(fig)
    return ax


def run_explanations(n_background: int = 3000):
    from src.models.train import load_splits

    pipeline = joblib.load(config.MODEL_PATH)
    meta = json.loads(config.METADATA_PATH.read_text())
    if meta["model_name"] not in ("XGBoost", "Random Forest"):
        print("  SHAP TreeExplainer needs a tree model - skipping.")
        return

    _, _, X_test, _, _, _ = load_splits()
    X_sample = X_test.sample(min(n_background, len(X_test)), random_state=config.RANDOM_STATE)

    exp = ReadmissionExplainer(pipeline)
    print(f"  Computing SHAP values for {len(X_sample):,} test patients...")
    grouped = exp.grouped_shap_values(X_sample)

    # 1. Global importance (grouped back to original features)
    plot_global_importance(grouped)

    # 2. Beeswarm on encoded features (shows direction of effect)
    X_enc = pd.DataFrame(exp.preprocessor.transform(X_sample[ALL_FEATURES]),
                         columns=exp.encoded_names)
    X_enc.columns = [c.split("__", 1)[1] for c in X_enc.columns]
    shap.summary_plot(exp.shap_values(X_sample), X_enc, max_display=15, show=False)
    plt.title("SHAP beeswarm: red = high feature value, blue = low", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "shap_beeswarm.png", dpi=130)
    plt.close()

    # 3. One high-risk patient, explained
    proba = pipeline.predict_proba(X_sample)[:, 1]
    idx = int(np.argsort(proba)[-25])            # a clearly high-risk example
    patient = X_sample.iloc[[idx]]
    table = exp.explain_patient(patient)
    plot_patient_explanation(table, proba[idx], filename="shap_patient_example.png")

    top = grouped.abs().mean().sort_values(ascending=False).head(10)
    print("  Top 10 risk drivers:")
    for f, v in top.items():
        print(f"    {READABLE_NAMES.get(f, f):<34} {v:.4f}")
    print(f"  SHAP plots saved -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    run_explanations()
