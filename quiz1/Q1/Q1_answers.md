# Quiz 1

## Q1a. What is open data?

Open data is data that anyone can freely access, use, modify, and share without restrictions, typically provided in machine-readable formats and under open licenses.

**Example:** Publishing local air-quality measurements (temperature, PM2.5, humidity) from personal sensors under a Creative Commons license so researchers or policymakers can reuse the data.

---

## Q1b. Handling Missing Attributes in a Dataset

### b.1) Possible Reasons for Missing Attributes
- **Data entry errors or omissions:** Human oversight, malfunctioning sensors, or corrupted files.
- **Privacy or confidentiality restrictions:** Sensitive fields intentionally withheld or removed.

### b.2) Ways to Proceed with Missing Data

| Approach | Assumption | Risk |
| --- | --- | --- |
| **Listwise deletion** | Missingness is completely at random (MCAR) | Loss of statistical power; bias if missingness is not random |
| **Mean/median imputation** | Missing values approximated by central tendency | Reduces variability; distorts relationships among variables |
| **Model-based imputation** (regression, k-NN, multiple imputation) | Missingness relates to other observed variables | Model misspecification can produce inaccurate imputations |
| **Algorithms tolerant to missing data** (e.g., decision trees) | Algorithm handles missingness internally | Risk of learning biased patterns or overfitting if mechanism is unknown |

