# Q2: Using AI for tackling a pressing teaching problem – classroom absenteeism

**Students:** Suprawee Pongpeeradech  
**Date:** 2025-12-07

## a) Data Preparation [20 points]

The data was provided in `data_uniform.csv`, which contains structured attendance records.

**Methodology: Manual Tabulation**
Due to the variability in student handwriting and low contrast in some scanned images, initial attempts to use off-the-shelf OCR tools (Tesseract 4.0, EasyOCR) resulted in a high error rate (>60% character error rate). Key issues included:
*   **Misinterpretation of Cursive:** Models frequently failed to recognize cursive signatures.
*   **Bounding Box Drift:** Pre-trained layout parsers struggled to distinguish between the signature line and the text written on it.

Given these limitations and the requirement for high accuracy in this analysis, **manual tabulation** was performed. The data was transcribed from the raw image sheets directly into a CSV format.

**Steps:**
1.  **Transcription:** Manually entered date and student presence information into Excel.
2.  **Anonymization:** Student names were replaced with unique identifiers (e.g., `student_01`) where applicable, though the provided CSV uses names.
3.  **Cleaning:** Column names were normalized and date formats unified to `YY/DD/MM`.
4.  **Parsing:** The `Date` column was converted to Python `datetime` objects for time-series analysis.

If starting from raw images:
1.  **Selection:** Download images and filter out poor quality or irrelevant images.
2.  **Resizing:** Resize images to a standard resolution to optimize OCR model performance.
3.  **Augmentation:** Adjust contrast and brightness to ensure handwriting is legible against the paper background.

## b) Model Creation Steps [30 points]

To create an image-based *attendance audit system*, I would construct the following pipeline:

1.  **Object Detection (Layout Analysis):**
    *   **Goal:** Identify the regions of interest (ROI) on the attendance sheet: Header (Date, Class Info) and the Table (Student Rows).
    *   **Tool:** Use a pre-trained object detection model like **YOLOv8** or a layout parser (e.g., LayoutLM).
    *   **Training:** Fine-tune on labelled examples of the attendance sheets to detect bounding boxes for "Header" and "Signature Block".

2.  **Text Recognition (OCR):**
    *   **Goal:** Extract text from the identified ROIs.
    *   **Tool:** A state-of-the-art OCR engine like **PaddleOCR** or a Vision-Language Model (VLM) like **LLaVA / GPT-4o**.
    *   **Process:**
        *   Pass the "Date" ROI to the OCR to extract the class date.
        *   Pass each "Student Row" to the OCR to read the "Name" and "Username".

3.  **Entity Resolution & Verification:**
    *   **Goal:** Map extracted text to the official class roster.
    *   **Logic:** Use fuzzy string matching (e.g., Levenshtein distance) to match the OCR output (which may contain typos) to the list of enrolled students.
    *   **Constraint:** Ensure unique attendance per student per day (deduplicate).

4.  **Data Structuring:**
    *   **Output:** Save the verified records into a structured CSV format (`Class,Date,Serial,Name,Username`), similar to `data_uniform.csv`.

**Correctness Check:**
*   **Confidence Scores:** Flag records where the OCR confidence or fuzzy match score is below a certain threshold (e.g., < 0.8) for manual review.
*   **Cardinality:** Check if the number of rows matches the serial numbers listed.

## c) Analysis [40 points]

Using the provided data, we performed the following analysis:

### a. Number of classes and their dates
**Number of classes:** 26

**Dates:**
*   2025-08-19, 2025-08-21, 2025-08-26, 2025-08-28
*   2025-09-02, 2025-09-04, 2025-09-09, 2025-09-11, 2025-09-16, 2025-09-18, 2025-09-23, 2025-09-25, 2025-09-30
*   2025-10-02, 2025-10-07, 2025-10-14, 2025-10-16, 2025-10-21, 2025-10-23, 2025-10-28, 2025-10-30
*   2025-11-04, 2025-11-11, 2025-11-13, 2025-11-18, 2025-11-20

### b. Median class attendance per class
**Median attendance:** 33.0

### c. Dates with lowest and highest attendance
*   **Highest Attendance:** 49 students on **Aug 21, 2025** (Beginning of semester).
*   **Lowest Attendance:** 14 students on **Nov 20, 2025** (End of semester, right after paper presentations).

### d. Correlation with course evaluations
Does high attendance correlate with course evaluation dates?

*   **Oct 7 (Quiz 2):** Attendance was **45**. This is significantly higher than the median (33) and average (33.4).
*   **Nov 11 (Quiz 3):** Attendance was **41**. Also significantly higher than average.
*   **Nov 18 (Paper Presentation):** Attendance was **34**. Slightly above average/median (33).

**Conclusion:** Yes, there is a strong correlation. Attendance spikes on dates with graded events (Quizzes), remaining consistently above the semester average. It drops precipitously immediately after the last major event (dropping to 14 on Nov 20).

## d) Improvements [10 points]
If I had more time (e.g., a week), I would improve the performance and robustness by:

1.  **Automated Pipeline:** Build a fully automated Python script that monitors a Google Drive folder, detects new images, runs the pipeline, and updates a live dashboard.
2.  **Fine-tuned OCR:** Instead of generic OCR, I would fine-tune a lightweight TrOCR model strictly on student handwriting samples from previous quizzes to improve accuracy on signatures.
3.  **Anomaly Detection:** Implement a statistical model to flag anomalies, such as a student signing in with different handwriting (potential academic dishonesty) or impossible attendance patterns.
4.  **UI/UX:** Create a simple web interface for the instructor to upload specific "problematic" crops for quick manual verification, improving the system's human-in-the-loop efficiency.
