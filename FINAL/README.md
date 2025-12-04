# CSCE 580 – Introduction to AI  
## Final Exam (Fall 2025)

**Student Name:** _Suprawee Pongpeeradech_  

This `readme.md` explains where each part of the **CSCE 580 Final Exam (Dec 4, 2025)** is located inside the `finals/` folder of this repository.

> 🔎 Reminder: You must complete **either Q1a OR Q1b**, not both. Delete the section you did **not** attempt, so it is clear to the graders.

Repository layout:
```text
finals/
  Q1a/
    Q1a_response.pdf        # or .md / .ipynb etc. (if you chose Q1a)
    code/                   # code for Q1a, if any
  Q1b/
    Q1b_response.pdf        # or .md / .ipynb etc. (if you chose Q1b)
    code/                   # code for Q1b, if any
  Q2/
    Q2_report.pdf           # main written report for Q2
    data/                   # (optional) processed/annotated data
    code/                   # notebooks, scripts, models for Q2
```

Update the file paths below so they exactly match what you actually committed.

---

## Q1 – Graduate Paper / Vocareum Lab (20 points)

### If you chose **Q1a: Graduate paper understanding**

- **Selected paper title:**  
  _<Write the full title of the paper here>_

- **Student presenter of that paper:**  
  _<Presenter's name here>_

- **Your written answers for Q1a (parts a–c) are in:**  
  - `./finals/Q1a/Q1a_response.pdf`  
    _or update this if you used a different filename or format (e.g., `.md`, `.ipynb`)._

- **Any code / experiments used for Q1a (if applicable):**  
  - `./finals/Q1a/code/`

Brief note (optional):
- _<One or two sentences describing what you did for the new example and analysis, if you like>_

---

### If you chose **Q1b: Vocareum traffic analysis notebook**

- **Vocareum notebook used:**  
  `Traffic-Analysis -> Files -> work -> chloropleth.ipynb` (on Vocareum)

- **Your written answers for Q1b (parts a–c) are in:**  
  - `./finals/Q1b/Q1b_response.pdf`

- **Final traffic visualization image (any one year as requested):**  
  - _Image file path in this repo:_ `./finals/Q1b/final_map_<year>.png`  
    (Change `<year>` and filename to match your actual image file.)

- **Any local copy / modified notebook or code used (optional):**  
  - `./finals/Q1b/code/`

Short checklist (you can keep or delete):
- [ ] Listed all data sources used in the notebook (part a).  
- [ ] Included at least one final SC traffic map image (part b).  
- [ ] Successfully ran the notebook on Vocareum (part c – instructor will verify backend).  

---

## Q2 – Image-based Attendance Audit System (80 points)

All work for Q2 is located in the `./finals/Q2/` directory.

### 1. Data preparation (Part a – 20 pts)

- **Description of data preparation and rationale is in:**  
  - `./finals/Q2/Q2_report.pdf` (Section: “Data preparation” / “Methodology”)

- **Any scripts / notebooks used to clean, resize, or preprocess the attendance images:**  
  - `./finals/Q2/code/preprocess_attendance.py`  
  - `./finals/Q2/code/attendance_preprocessing.ipynb`  
  _(Rename / update to match your actual filenames.)_

- **Optional processed data or intermediate tables (if stored):**  
  - `./finals/Q2/data/`

### 2. Modeling / Extraction approach (Part b – 30 pts)

Here you explain how you extracted names/usernames, dates, class numbers, etc., from the images.

- **Written explanation (pre‑trained models, your own model, or manual approach):**  
  - `./finals/Q2/Q2_report.pdf` (Section: “Modeling / Extraction approach”)

- **Model / code paths (choose the ones that apply):**
  - Pre‑trained model scripts / notebooks:  
    - `./finals/Q2/code/llm_extraction.ipynb`  
    - `./finals/Q2/code/yolo_attendance_detector.ipynb`
  - Manual or semi‑manual tabulation scripts:  
    - `./finals/Q2/code/manual_tabulation.ipynb`
  - Any saved model weights or configs (if small enough to include):  
    - `./finals/Q2/code/models/`

### 3. Attendance analysis & answers (Part c – 40 pts total)

Your final numerical answers and analysis should be clearly visible in the report.

- **Main analysis & answers for all subparts (c.a–c.d):**  
  - `./finals/Q2/Q2_report.pdf` (Section: “Attendance analysis and results”)

Suggested structure inside your report:

1. **Number of classes and their dates (Part c.a – 10 pts)**  
   - Table or list showing: `Class #`, `Date`, and maybe `Day of week`.

2. **Median class attendance per class (Part c.b – 10 pts)**  
   - Table of attendance counts per class.  
   - Median value clearly stated.

3. **Dates with lowest and highest attendance (Part c.c – 10 pts)**  
   - Explicitly list:  
     - Lowest attendance date(s) + value  
     - Highest attendance date(s) + value  

4. **Correlation with course evaluation dates (Part c.d – 10 pts)**  
   - Discuss attendance around:  
     - Quiz 2 – Oct 7  
     - Quiz 3 – Nov 11  
     - Paper presentations – Nov 18  
   - Clearly state: “When is the attendance highest?” and relate it to the evaluation dates.

If you have extra plots / tables:

- Attendance plots, histograms, or time‑series graphs:  
  - `./finals/Q2/figures/attendance_over_time.png`  
  - `./finals/Q2/figures/attendance_vs_evaluations.png`

### 4. Possible improvements (Part d – 10 pts)

- **Your discussion of “what more you would do with one extra week” is in:**  
  - `./finals/Q2/Q2_report.pdf` (Section: “Future work / Possible improvements”)
