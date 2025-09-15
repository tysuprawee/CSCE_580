# Q3 – Fire call analysis

This README documents the answers for Question 3 of the quiz. Results were produced with the helper script `analyze.py` which keeps the original dataset untouched while performing all cleaning and computations.

## Question a – Data issues

### a1. Data range
- Dispatch records span **24 March 2025, 15:54** through **31 August 2025, 23:03**.
- After correcting the completion timestamps, the calls close between **24 March 2025, 16:20** and **31 August 2025, 23:05**.

### a2. Missing data
| Column | Missing rows |
| --- | --- |
| `1ST UNIT ON SCENE` | 428 |
| `SHIFT` | 69 |
| `ALARM DATE TIME` | 31 |
| `CALL COMPLETE` | 31 |

### a3. Issues discovered
- **Completion dates disconnected from dispatch dates.** Every `CALL COMPLETE` entry was stamped as either 4 or 5 September 2025, which inflated raw call durations by multiple months. The `cleaned_call_complete` helper replaces the incorrect calendar date with the dispatch date while keeping the recorded time-of-day; when the time appears earlier than dispatch it is rolled into the next day.【F:quiz1/Q3/analyze.py†L27-L57】【F:quiz1/Q3/analyze.py†L233-L242】
- **Alarm timestamps with inconsistent days.** Several alarms reused the correct time-of-day but a different day or month, creating implausible gaps of days or weeks. The `alarm_baseline` logic realigns anomalous alarms to the dispatch date whenever the difference exceeds 12 hours, yielding realistic response gaps.【F:quiz1/Q3/analyze.py†L59-L83】
- **Duplicate incident identifiers.** Incident `25-2023` appears twice; downstream reporting should deduplicate or aggregate those entries.【F:quiz1/Q3/analyze.py†L129-L133】
- **Multi-value dispatch units.** Units are stored as comma-separated strings. The loader splits them into a list so unit counts can be analysed reliably.【F:quiz1/Q3/analyze.py†L105-L116】【F:quiz1/Q3/analyze.py†L186-L205】

### a4. Cleaning outcomes
- The worst case highlighted above now resolves in 26 minutes instead of five months because the completion timestamp is projected onto the dispatch day.【F:quiz1/Q3/analyze.py†L27-L57】【F:quiz1/Q3/analyze.py†L233-L242】【F:quiz1/Q3/analyze.py†L245-L252】
- Alarms that were off by a day are realigned, reducing exaggerated delays to zero minutes when appropriate.【F:quiz1/Q3/analyze.py†L59-L83】【F:quiz1/Q3/analyze.py†L209-L214】

### a5. Post-cleaning quality check
- Call resolution now averages **138.6 minutes** (median **12.0 minutes**), with a maximum of **1,439 minutes**—well within a 24-hour window.【F:quiz1/Q3/analyze.py†L141-L168】
- Alarm-to-dispatch gaps centre on **6.0 minutes** (median **4.0 minutes**), confirming short dispatch delays after realignment.【F:quiz1/Q3/analyze.py†L170-L178】
- Alarm-to-close durations average **144.2 minutes**, demonstrating consistency between the cleaned timelines.【F:quiz1/Q3/analyze.py†L170-L178】

## Question b – Exploratory data analysis

### b1. Average time from call creation to closure
Calls resolve in **138.6 minutes on average** (median **12.0 minutes**) after the cleaning step described above.【F:quiz1/Q3/analyze.py†L141-L168】

### b2. Time from alarm to closure
Interpreting `ALARM DATE TIME` as the earliest recorded event, the window from alarm to final clearance averages **144.2 minutes** with a **17.0 minute** median.【F:quiz1/Q3/analyze.py†L170-L178】 This aligns with the short dispatch gaps observed in part (a5).

### b3. Busiest shift
Shift **A** handled **735** incidents during the study window, slightly ahead of shift C (719) and shift B (677).【F:quiz1/Q3/analyze.py†L180-L188】 The 69 uncategorised rows highlight the need for better shift logging.

### b4. Incidents by weekday and hour
The tables below show call counts by weekday and hour of dispatch. Totals across both halves equal the 2,200 logged incidents.【F:quiz1/Q3/analyze.py†L190-L216】

Segment 00–11

| Day/Hour | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Monday | 8 | 11 | 4 | 9 | 4 | 9 | 4 | 15 | 13 | 11 | 17 | 24 |
| Tuesday | 8 | 8 | 5 | 10 | 4 | 3 | 5 | 15 | 13 | 18 | 22 | 20 |
| Wednesday | 6 | 7 | 3 | 8 | 4 | 8 | 7 | 8 | 18 | 14 | 18 | 15 |
| Thursday | 2 | 8 | 3 | 1 | 2 | 4 | 7 | 11 | 8 | 14 | 16 | 25 |
| Friday | 3 | 3 | 4 | 9 | 7 | 6 | 10 | 14 | 8 | 14 | 16 | 18 |
| Saturday | 7 | 8 | 8 | 4 | 6 | 4 | 6 | 5 | 7 | 10 | 18 | 15 |
| Sunday | 4 | 10 | 8 | 9 | 4 | 5 | 10 | 10 | 14 | 14 | 11 | 12 |
| Total | 38 | 55 | 35 | 50 | 31 | 39 | 49 | 78 | 81 | 95 | 118 | 129 |

Segment 12–23

| Day/Hour | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Monday | 16 | 18 | 22 | 20 | 24 | 28 | 19 | 20 | 17 | 16 | 3 | 12 |
| Tuesday | 15 | 13 | 27 | 23 | 17 | 16 | 22 | 12 | 11 | 16 | 12 | 8 |
| Wednesday | 16 | 13 | 18 | 25 | 14 | 17 | 27 | 20 | 21 | 15 | 18 | 9 |
| Thursday | 20 | 15 | 21 | 25 | 21 | 12 | 22 | 17 | 17 | 13 | 8 | 8 |
| Friday | 29 | 11 | 24 | 17 | 17 | 28 | 15 | 18 | 14 | 14 | 13 | 9 |
| Saturday | 13 | 18 | 16 | 17 | 14 | 17 | 16 | 17 | 23 | 13 | 13 | 29 |
| Sunday | 18 | 20 | 12 | 20 | 11 | 20 | 13 | 11 | 11 | 7 | 16 | 9 |
| Total | 127 | 108 | 140 | 147 | 118 | 138 | 134 | 115 | 114 | 94 | 83 | 84 |

## Question c – Unsupervised learning

### c1. Clustering methods and quality
- **K-means (k = 3)** on six engineered features (call duration, alarm-to-dispatch gap, alarm-to-close duration, dispatch hour, units dispatched, shift code) achieved a **silhouette score of 0.304** with inertia **6,968** and cluster sizes {622, 1,327, 182}.【F:quiz1/Q3/analyze.py†L218-L242】
- **DBSCAN** with ε = 1.25 and `min_samples` = 25 produced four clusters plus 104 noisy points, but the silhouette dropped to **0.239**.【F:quiz1/Q3/analyze.py†L244-L252】

K-means therefore offered clearer separation for this dataset.

### c2. Cluster interpretation
Using the better-performing K-means model, cluster centroids (converted back to the original units) reveal three behavioural groups:
- **Cluster 2 (n = 1,327)** – short engagements averaging **13.9 minutes** with a single responding unit; these are routine alarms mostly on shift B.【F:quiz1/Q3/analyze.py†L186-L205】【6e6f1b†L1-L4】
- **Cluster 1 (n = 622)** – slightly longer responses (**20.2 minutes**) that dispatch an average of 2.4 units, representing multi-unit interventions.【6e6f1b†L1-L3】
- **Cluster 0 (n = 182)** – protracted incidents lasting roughly **24 hours**; they feature longer alarm gaps and likely correspond to extended operations or problem data that merit manual review.【6e6f1b†L3-L4】

These segments can guide staffing: most calls are fast and single-unit, while a small fraction requires full-shift commitments.
