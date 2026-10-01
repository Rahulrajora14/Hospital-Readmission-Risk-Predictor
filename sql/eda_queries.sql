-- =====================================================================
--  Exploratory Data Analysis in SQL
--  Table: encounters (one row = one hospital visit of a diabetic patient)
--
--  Each query starts with "-- name: <query_name>". The Python runner
--  (src/data/sql_eda.py) splits this file on those markers, runs every
--  query, prints the result and saves it to reports/sql_results/.
--
--  readmitted column values:
--     '<30' -> readmitted within 30 days   (our positive class)
--     '>30' -> readmitted after 30 days
--     'NO'  -> not readmitted
-- =====================================================================


-- name: 01_dataset_overview
-- How big is the data and how many unique patients are there?
SELECT
    COUNT(*)                                             AS total_encounters,
    COUNT(DISTINCT patient_nbr)                          AS unique_patients,
    ROUND(100.0 * SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_readmitted_30d,
    ROUND(100.0 * SUM(CASE WHEN readmitted = '>30' THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_readmitted_after_30d,
    ROUND(100.0 * SUM(CASE WHEN readmitted = 'NO' THEN 1 ELSE 0 END)  / COUNT(*), 2) AS pct_not_readmitted
FROM encounters;


-- name: 02_missing_values
-- The dataset marks missing values with '?'. Which columns are worst?
SELECT 'weight'            AS column_name, ROUND(100.0 * SUM(CASE WHEN weight = '?' THEN 1 ELSE 0 END) / COUNT(*), 1)            AS pct_missing FROM encounters
UNION ALL
SELECT 'medical_specialty',               ROUND(100.0 * SUM(CASE WHEN medical_specialty = '?' THEN 1 ELSE 0 END) / COUNT(*), 1) FROM encounters
UNION ALL
SELECT 'payer_code',                      ROUND(100.0 * SUM(CASE WHEN payer_code = '?' THEN 1 ELSE 0 END) / COUNT(*), 1)        FROM encounters
UNION ALL
SELECT 'race',                            ROUND(100.0 * SUM(CASE WHEN race = '?' THEN 1 ELSE 0 END) / COUNT(*), 1)              FROM encounters
ORDER BY pct_missing DESC;


-- name: 03_readmission_by_age
-- Does readmission risk change with age?
SELECT
    age                                                  AS age_group,
    COUNT(*)                                             AS encounters,
    ROUND(100.0 * SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) / COUNT(*), 2) AS readmit_30d_rate_pct
FROM encounters
GROUP BY age
ORDER BY age;


-- name: 04_readmission_by_prior_inpatient_visits
-- Patients who were admitted many times last year: are they riskier?
SELECT
    CASE
        WHEN number_inpatient = 0 THEN '0'
        WHEN number_inpatient = 1 THEN '1'
        WHEN number_inpatient = 2 THEN '2'
        ELSE '3+'
    END                                                  AS prior_inpatient_visits,
    COUNT(*)                                             AS encounters,
    ROUND(100.0 * SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) / COUNT(*), 2) AS readmit_30d_rate_pct
FROM encounters
GROUP BY prior_inpatient_visits
ORDER BY prior_inpatient_visits;


-- name: 05_readmission_by_a1c_result
-- HbA1c is the key diabetes blood test. Does testing it matter?
SELECT
    A1Cresult                                            AS a1c_result,
    COUNT(*)                                             AS encounters,
    ROUND(100.0 * SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) / COUNT(*), 2) AS readmit_30d_rate_pct
FROM encounters
GROUP BY A1Cresult
ORDER BY encounters DESC;


-- name: 06_readmission_by_length_of_stay
-- Longer stays usually mean sicker patients.
SELECT
    time_in_hospital                                     AS days_in_hospital,
    COUNT(*)                                             AS encounters,
    ROUND(100.0 * SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) / COUNT(*), 2) AS readmit_30d_rate_pct
FROM encounters
GROUP BY time_in_hospital
ORDER BY time_in_hospital;


-- name: 07_readmission_by_discharge_destination
-- Where the patient went after discharge (top 8 destinations).
-- IDs 11, 19, 20, 21 = patient expired; 13, 14 = hospice.
SELECT
    discharge_disposition_id,
    COUNT(*)                                             AS encounters,
    ROUND(100.0 * SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) / COUNT(*), 2) AS readmit_30d_rate_pct
FROM encounters
GROUP BY discharge_disposition_id
ORDER BY encounters DESC
LIMIT 8;


-- name: 08_repeat_patients
-- Same patient can appear many times. If we keep all their visits,
-- one visit could land in train and another in test = data leakage.
-- This query uses a CTE to count how many patients repeat.
WITH visits_per_patient AS (
    SELECT patient_nbr, COUNT(*) AS n_visits
    FROM encounters
    GROUP BY patient_nbr
)
SELECT
    CASE WHEN n_visits = 1 THEN '1 visit'
         WHEN n_visits <= 3 THEN '2-3 visits'
         ELSE '4+ visits' END                            AS visit_bucket,
    COUNT(*)                                             AS patients,
    SUM(n_visits)                                        AS encounters
FROM visits_per_patient
GROUP BY visit_bucket
ORDER BY visit_bucket;


-- name: 09_first_visit_per_patient
-- Window function: keep only each patient's FIRST encounter.
-- This is exactly the de-duplication rule the Python cleaning step uses.
WITH ranked AS (
    SELECT
        patient_nbr,
        readmitted,
        ROW_NUMBER() OVER (PARTITION BY patient_nbr ORDER BY encounter_id) AS visit_rank
    FROM encounters
)
SELECT
    COUNT(*)                                             AS first_visits,
    ROUND(100.0 * SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) / COUNT(*), 2) AS readmit_30d_rate_pct
FROM ranked
WHERE visit_rank = 1;


-- name: 10_top_specialties_by_risk
-- Admitting specialties with at least 500 encounters, ranked by risk.
SELECT
    medical_specialty,
    COUNT(*)                                             AS encounters,
    ROUND(100.0 * SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) / COUNT(*), 2) AS readmit_30d_rate_pct
FROM encounters
WHERE medical_specialty <> '?'
GROUP BY medical_specialty
HAVING COUNT(*) >= 500
ORDER BY readmit_30d_rate_pct DESC;
