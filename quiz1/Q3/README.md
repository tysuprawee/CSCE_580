# Fire call analysis (Quiz 1 – Question 3)

This folder contains scripts and artefacts for analysing the Midlands fire
station call logs provided for Quiz 1, Question 3.  The dataset is stored at
`data/data.csv` and consists of 2,200 dispatch records covering March–May 2025.

## Data quality findings

Running `python q3_analysis.py` surfaces several quality issues that must be
addressed before analysis:

- **Missing values.** `1ST UNIT ON SCENE` is blank in 428 rows (19.45%), `SHIFT`
  is absent in 69 rows (3.14%), and 31 rows (1.41%) lack both `ALARM DATE TIME`
  and `CALL COMPLETE` timestamps.
- **Duplicate incident numbers.** Incident `25-2023` appears twice, suggesting
  multi-dispatch handling for a single incident number.
- **Inconsistent formatting.** Dispatch unit identifiers vary in whitespace and
  hyphenation (for example `E171` versus `E-171`), which prevents reliable
  grouping until normalised.
- **Long administrative closures.** Calculated resolution times (dispatch to
  `CALL COMPLETE`) span 4–165 days because calls remain open administratively
  long after on-scene activity. This limits the metric's usefulness for
  operational response analysis but is reported for completeness.

## Cleaning strategy

The script follows a consistent approach to resolve these issues:

1. Strip whitespace from every field and normalise dispatch unit identifiers to
   uppercase without embedded spaces.
2. Parse dispatch, alarm, and completion timestamps into Python `datetime`
   objects; rows without a dispatch timestamp are excluded because timing
   metrics depend on that field.
3. Add a sequential `case_id` to each record and preserve the original
   `xref_id`; all cleaning occurs in-memory only, leaving the CSV on disk
   untouched.
4. Replace missing categorical values with the label `UNKNOWN` so that counts
   remain explicit. No numeric imputation is performed; rows lacking timestamps
   are simply omitted from calculations that require them.
5. Derive helper columns such as unit counts, response durations (in hours),
   ISO week numbers, and hour-of-day indicators to support the exploratory
   questions.

## Exploratory highlights

- **Average resolution time:** 2,176.78 hours from dispatch to `CALL COMPLETE`.
- **Average units per call:** 1.44 vehicles.
- **Busiest shift:** Shift A handled 735 calls, slightly ahead of Shift C (719)
  and Shift B (677). The 69 `UNKNOWN` entries reflect missing shift codes.
- **Weekly/hourly grid:** The script identifies the busiest ISO week/hour
  combination (week 18 at 15:00 with 14 calls) and reports the total number of
  records containing sufficient timestamp detail. No additional CSV artefacts
  are produced.

## How to run

Execute the following command from the repository root to print the summary
statistics:

```bash
python quiz1/Q3/q3_analysis.py
```
