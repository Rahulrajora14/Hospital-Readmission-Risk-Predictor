# 📘 How this project works

A complete walkthrough of the codebase: what each file does, why each decision was made, and how to explain it in an interview. Read it top to bottom once, then use it as a reference.

---

## 1. The big picture in one minute

**Question the project answers:** *"When a diabetic patient leaves hospital today, how likely are they to be back within 30 days, and why?"*

**The flow:**

```
CSV file ─► SQLite DB ─► SQL analysis
                │
                └─► clean ─► features ─► split ─► train 3 models ─► pick threshold
                                                       │
                                                       ▼
                                              best model saved
                                              │            │
                                         SHAP explain   Streamlit app
```

**One command runs it all:** `python main.py`. Each arrow above is one Python function, and `main.py` simply calls them in order.

**How to describe it in 30 seconds (memorise this):**

> "I built an end-to-end readmission risk model on 100k real US hospital records. I loaded the data into SQL and did the EDA with queries, then cleaned it with leakage rules: one visit per patient and no deceased or hospice patients. I engineered clinical features like ICD-9 disease groups and medication-change counts, compared Logistic Regression, Random Forest and an Optuna-tuned XGBoost, and handled the 9% class imbalance by tuning the decision threshold on a validation set. The final model has 2.2× lift in the top risk decile, and I used SHAP to explain every prediction in a Streamlit dashboard."

---

## 2. File-by-file walkthrough

Read the files in this order; it follows the pipeline.

### `src/config.py`: the control panel

Every path (where the data lives, where models are saved) and every setting (random seed, test size, number of tuning trials) is defined here once. Other files import from it: `from src import config`.

**Why:** if you hard-code `"data/raw/file.csv"` in five files and later move it, you have to fix five files. With a config file you fix one line. `ROOT = Path(__file__).resolve().parents[1]` finds the project folder automatically, so the code runs on any machine.

---

### `src/data/download_data.py`: step 1

Downloads the zip from UCI and extracts `diabetic_data.csv`. If UCI is down, it tries a GitHub mirror. If the file already exists, it skips.

**Key idea:** a pipeline should be re-runnable without side effects ("idempotent").

---

### `src/data/load_to_sql.py`: step 2

Reads the CSV with pandas and writes it into a SQLite database table called `encounters` using `df.to_sql(...)`. It also creates two **indexes** so `GROUP BY patient_nbr` queries run fast.

- `keep_default_na=False` keeps the dataset's own markers (`'?'`, `'None'`) as text, so the database matches the source exactly. We decide later, in cleaning, what they mean.
- `read_table(query)` is a helper: give it any SQL, get a DataFrame back. The cleaning step uses it, so **the ML pipeline reads its data through SQL**, just like at a company.

**Why SQLite:** no server to install. The whole database is one file, `data/readmission.db`. The SQL is standard, so the same queries work on PostgreSQL or Databricks SQL.

---

### `sql/eda_queries.sql` + `src/data/sql_eda.py`: step 3

The `.sql` file holds 10 queries, each starting with `-- name: something`. The Python runner splits the file on those markers, runs each query, prints it, saves it as CSV in `reports/sql_results/`, and draws three bar charts.

SQL techniques you should be able to explain:

| Technique | Where | What it does |
|---|---|---|
| `SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END)` | almost every query | counts rows that match a condition; dividing by `COUNT(*)` gives a rate |
| `GROUP BY` + `ORDER BY` | queries 03–07 | readmission rate per age group, per stay length, etc. |
| `CASE` bucketing | query 04 | groups 0 / 1 / 2 / 3+ prior visits |
| `UNION ALL` | query 02 | stacks several one-row results into a table |
| `HAVING` | query 10 | filters **groups** (specialties with 500+ rows); `WHERE` filters rows |
| **CTE** (`WITH ... AS`) | queries 08–09 | a named temporary result you can query again, which reads better than nested sub-queries |
| **Window function** `ROW_NUMBER() OVER (PARTITION BY patient_nbr ORDER BY encounter_id)` | query 09 | numbers each patient's visits 1, 2, 3... so `WHERE visit_rank = 1` keeps the first visit |

**What the EDA told us (and how it shaped the model):**

- Positive class is only ~9–11% → imbalance → do not use accuracy (see section 3).
- `weight` is 97% missing → dropped.
- Prior inpatient visits strongly raise risk → a strong feature, and it motivated `total_prior_visits`.
- 16,773 patients have multiple visits → must de-duplicate (leakage).
- Discharge IDs 11/19/20/21 mean the patient died → must remove.

---

### `src/features/clean.py`: step 4a

Applies five rules, printing row counts after each:

1. `'?'` → `NaN` (real missing values).
2. Remove `discharge_disposition_id` in `[11, 13, 14, 19, 20, 21]` (expired or hospice). These patients **cannot** be readmitted, so leaving them in teaches the model "this type of patient never comes back", which is false.
3. Sort by `encounter_id`, keep the **first** row per `patient_nbr`. This is the most important line in the project. Without it, the same person can be in train and test, and the model partly memorises people instead of learning patterns.
4. Remove `Unknown/Invalid` gender (3 rows).
5. Target: `readmitted_30d = 1` if `readmitted == '<30'`, else 0. Note that `'>30'` counts as **negative**: the business question is about 30 days.

---

### `src/features/engineer.py`: step 4b

Defines **the exact feature lists** the model uses (`NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`) and builds them.

| Transformation | Code idea | Reasoning |
|---|---|---|
| ICD-9 grouping | `map_icd9_to_group("428") → "Circulatory"` | 700+ codes would create 700 sparse columns; 9 medically meaningful groups generalise better |
| Age to number | `"[70-80)" → 75` | a number keeps order (75 > 65), a category would not |
| ID lookups | `ADMISSION_TYPE = {1: "Emergency", ...}` | IDs like `6` mean nothing to a model or a human; groups merge rare IDs |
| Top-N specialty | keep 10 most common, rest `"Other"` | limits cardinality, avoids overfitting rare categories |
| "Not measured" lab | `'None' → 'Not measured'` | not running a test is a clinical decision, so it is information, not noise |
| `num_meds_changed` | count of 23 med columns equal to `Up` or `Down` | dose changes signal unstable diabetes |
| `num_meds_active` | count of med columns not equal to `No` | treatment complexity |
| `total_prior_visits` | outpatient + emergency + inpatient | overall utilisation |

**Race is intentionally not a feature** (fairness; see section 6).

### `src/features/build_dataset.py`

Glue code: `read_table()` (SQL) → `clean_data()` → `engineer_features()` → save `data/processed/clean_data.csv` (69,987 rows × 27 columns: 26 features + target).

---

### `src/models/train.py`: step 5 (the heart of the project)

**1. `load_splits()`**: two calls to `train_test_split` give 70% train, 15% validation, 15% test. `stratify=y` makes every split keep the ~9% positive rate. `random_state=42` makes the split identical every run.

*Why three sets?*
- **Train:** the model learns from it.
- **Validation:** used for decisions (tuning, threshold, picking the best model).
- **Test:** used **once** at the end. If you make decisions using test scores, the test score is no longer an honest estimate.

**2. `build_preprocessor()`**: a `ColumnTransformer` that applies:
- `StandardScaler` to numbers (only for Logistic Regression; it uses gradient-based optimisation and regularisation, which are sensitive to feature scale. Trees split on thresholds, so scale does not matter).
- `OneHotEncoder(handle_unknown="ignore", min_frequency=30)` to categories. `handle_unknown="ignore"` means a new category at prediction time becomes all zeros instead of crashing. `min_frequency=30` merges very rare categories.

**3. `Pipeline([preprocess, model])`**: preprocessing and model are glued into one object. Benefits: the scaler/encoder are fitted **only on training data** (no leakage), and the saved `.joblib` file contains everything, so the app passes raw values and gets a prediction.

**4. The three models:**

| Model | Idea in one line | Role here |
|---|---|---|
| Logistic Regression | weighted sum of features → sigmoid → probability | baseline; if complex models cannot beat it, they are not worth it |
| Random Forest | many deep trees on random subsets of rows/columns, averaged (**bagging**) | reduces variance, captures interactions |
| XGBoost | trees built **one after another**, each fixing the errors of the previous ones (**boosting**), with regularisation | usually the strongest on tabular data |

**5. `tune_xgboost()` with Optuna:** each trial picks values for `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, trains on train, scores PR-AUC on validation. Optuna's **TPE sampler** builds a probability model of which parameter regions gave good scores and samples more from there. It is smarter than grid search (tries every combination) and random search (ignores past results).

What the main hyperparameters do:
- `learning_rate`: how big each correction step is (small = slower but more careful; needs more trees)
- `max_depth`: how complex each tree can be (deeper = more overfitting risk)
- `min_child_weight`: minimum data in a leaf (higher = more conservative)
- `subsample` / `colsample_bytree`: train each tree on a random fraction of rows / columns (adds randomness, reduces overfitting)
- `reg_lambda`: L2 penalty on leaf weights

**6. Threshold selection (imbalance handling):** a model outputs a probability, such as 0.14. To say "flag this patient", we need a cut-off. With 9% positives, the default 0.5 would flag almost nobody. `find_best_threshold()` tries every possible cut-off on the **validation** set and picks the one with the highest F1. For XGBoost it is about **0.126**.

*Why not SMOTE or class weights?* They also work, but they distort the predicted probabilities (the model starts believing positives are common). Threshold moving keeps probabilities realistic, so "14%" in the app roughly means 14%. Good interview point: you know several methods and chose one deliberately.

**7. Model selection:** the best **validation** PR-AUC wins (XGBoost). Then it is saved with `joblib.dump`, plus `model_metadata.json` (threshold, metrics, feature lists, category options and defaults the app needs).

**8. MLflow:** `log_to_mlflow()` records each model's parameters and metrics in `mlflow.db`. Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to compare runs in a browser. It is wrapped in `try/except` so tracking problems never break training.

---

### `src/models/evaluate.py`: metrics and plots

| Metric | Plain meaning | Our XGBoost |
|---|---|---|
| **Accuracy** ❌ | % correct. Useless here: predicting "no" for everyone scores 91% | not used |
| **ROC-AUC** | pick one readmitted and one not-readmitted patient at random; probability the model ranks the readmitted one higher. 0.5 = coin flip | 0.646 |
| **PR-AUC** (average precision) | area under precision-recall curve; focuses on the rare class. Baseline = positive rate (0.09) | 0.164 (≈1.8× baseline) |
| **Recall** | of all real readmissions, how many we caught | 35% |
| **Precision** | of the patients we flagged, how many were really readmitted | 18% |
| **F1** | harmonic mean of precision and recall | 0.239 |
| **Top-10% lift** | readmission rate in the 10% highest-scored patients ÷ overall rate | 2.22× |

**Top-10% lift is the business metric.** A care team with capacity for 10% of patients gets a list where ~20% will be readmitted instead of ~9%. That is how you would pitch it to Evernorth/Cigna leadership.

The confusion matrix (test set, threshold 0.126): 330 readmissions caught, 613 missed, 1,489 false alarms, 8,067 correctly cleared. In healthcare a false alarm costs a phone call; a missed readmission costs a hospital stay. That trade-off is why we accept low precision for better recall.

---

### `src/explain/shap_explainer.py`: step 6

**SHAP in one sentence:** for each prediction, SHAP splits the model's output into a contribution from every feature, so `base value + all contributions = this patient's score` (in log-odds).

- **Positive** contribution → pushes risk up (red in our charts)
- **Negative** → pushes risk down (teal)

`TreeExplainer` computes exact SHAP values efficiently for tree models.

`ReadmissionExplainer` class:
- `shap_values(X)`: SHAP per **encoded** column (`cat__diag_1_group_Circulatory`, ...).
- `grouped_shap_values(X)`: adds the one-hot columns back together per original feature. This is valid because SHAP values are **additive**. Result: one number for "Primary diagnosis" instead of nine.
- `explain_patient(row)`: top 8 factors for one patient. The app uses this.

Outputs: global importance bar chart, beeswarm (each dot is a patient; colour = feature value, x-position = impact), and a single-patient chart.

**Reading the beeswarm:** for `number_inpatient`, red dots (many prior stays) sit far right (higher risk). For `discharge_group_Home`, red (= yes, went home) sits left (lower risk).

---

### `app/streamlit_app.py`: the dashboard

- `@st.cache_resource` loads the model, metadata and explainer **once**, not on every click.
- Every input widget uses the **feature name as its key**, so "Load a real test patient" can fill the whole form by writing to `st.session_state`.
- On every change, the app builds a one-row DataFrame, calls `pipeline.predict_proba`, compares with the saved threshold and draws the SHAP chart.
- `result_area = st.container()` is created above the form but filled after it, so the result shows at the top.
- `sys.path.insert(0, ROOT)` lets the app import `src` when Streamlit runs the file directly.

---

### `main.py`, `tests/`, CI

- `main.py`: a dictionary `STEPS` maps step names to functions. `argparse` lets you run all steps, one step (`--step`), or from a step onward (`--from`).
- `tests/test_features.py`: `pytest` unit tests on tiny hand-made data. They check the ICD-9 mapping, that leakage rules remove the right rows, and that features are computed correctly. `@pytest.mark.parametrize` runs one test function on many inputs.
- `.github/workflows/tests.yml`: GitHub Actions runs the tests automatically on every push (a green tick on your repo).

---

## 3. Why results are "only" ~0.65 ROC-AUC, and why that is fine

Interviewers may ask this. Strong answer:

1. **Published, leakage-free work on this dataset reports similar ROC-AUC (~0.63–0.68).** Readmission depends heavily on things not in the data: home support, whether the patient takes their medicine, social circumstances.
2. **Many online notebooks report 0.90+ because of leakage**: keeping repeat patients, oversampling before splitting, or treating `>30` as positive (an easier, different problem). I deliberately avoided all of these.
3. **The model is still useful:** 2.2× lift in the top decile means a care-management team's calls are more than twice as productive.
4. **Honest evaluation matters more than a big number.** In healthcare, a model that looks great offline and fails in production is dangerous.

---

## 4. How to experiment (good for learning)

| Try this | Where | What you will learn |
|---|---|---|
| Set `N_TUNING_TRIALS = 60` | `config.py` | whether more tuning helps (usually a little) |
| Comment out the first-visit rule | `clean.py` | watch test scores jump; that is leakage in action |
| Add `scale_pos_weight=10` to XGBoost | `train.py` | class weighting vs threshold moving |
| Add a feature such as `num_medications / time_in_hospital` | `engineer.py` + feature list | feature engineering impact |
| Change threshold strategy to target recall ≥ 0.5 | `evaluate.find_best_threshold` | precision/recall trade-off |

After any change: `python main.py --from clean`, then compare in MLflow.

---

## 5. Interview questions you should be ready for

**Q: Why did you remove repeat patients?**
If the same patient appears in train and test, the model can recognise them instead of learning general patterns, so the test score is inflated. Keeping the first visit per patient guarantees each person is in only one split.

**Q: Why not accuracy?**
Only 9% are positive. Always predicting "not readmitted" gives 91% accuracy and is useless. I used ROC-AUC, PR-AUC, recall, precision, F1 and top-decile lift.

**Q: ROC-AUC vs PR-AUC?**
ROC-AUC measures ranking across both classes and can look decent even when the positive class is hard to find. PR-AUC focuses only on how well we find positives, so it is more informative for rare events. Its random baseline equals the positive rate.

**Q: How did you handle imbalance?**
Stratified splits, imbalance-aware metrics, and threshold moving: I chose the decision threshold that maximised F1 on validation. I considered SMOTE and class weights but kept threshold moving because it preserves calibrated probabilities.

**Q: Why XGBoost over Random Forest?**
Boosting builds trees sequentially to correct previous errors, which usually wins on tabular data. Here it had the best validation PR-AUC; on test the two were nearly tied, so in production I would also weigh speed and simplicity.

**Q: What does Optuna do better than GridSearch?**
GridSearch tries every combination blindly. Optuna's TPE sampler learns from previous trials which regions look good and samples more there, so it finds good settings with fewer trials.

**Q: Explain SHAP to a non-technical stakeholder.**
For each patient, SHAP shows how much each factor pushed the risk up or down from the average, like an itemised bill for the score.

**Q: Why exclude race?**
To avoid the model encoding historical inequalities in care. Next I would audit performance across groups (fairness check).

**Q: How would you deploy this?**
Wrap the saved pipeline in a FastAPI endpoint, containerise with Docker, schedule retraining, and monitor data drift and model performance. On Databricks I would use MLflow's model registry.

**Q: What would improve the model most?**
Better data rather than a fancier model: social and adherence information, lab values over time, and the full visit history modelled properly (with a patient-level split instead of first-visit only).

**Q: What is data leakage? Give two examples from your project.**
Information available during training that would not be available at prediction time, or that ties train and test together. Examples: repeat patients across splits, and deceased patients (outcome determined by something other than readmission risk).

**Q: Why put the encoder inside the Pipeline?**
So it is fitted only on training data and saved with the model. The same transformation is applied at prediction time, with no train/serve mismatch.

---

## 6. Responsible AI notes

- Race excluded from features.
- The model supports decisions (who to call first); it does not make them.
- 1999–2008 US data: retrain and re-validate before using elsewhere.
- Not clinically validated; educational project.
