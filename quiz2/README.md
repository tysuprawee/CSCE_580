# CSCE 580 — Quiz 2 (October 7, 2025)

This repository contains my submission for **CSCE 580: Introduction to AI — Quiz 2**.
The work captures energy comparisons for AI tooling, two recipe-conversion
experiments, optional prompting explorations, and rubric-based evaluation
utilities.

## Repository layout

```
Quiz-2/
├── code/   # Python helpers for combining partial prompts, scoring JSON, and GAICO-style comparison
├── data/   # Original recipe text plus generated R3 JSON outputs (PF, PP, optional experiments)
└── docs/   # Written responses and commentary (answer-comments.md)
```

## Reproducing the evaluation

1. (Optional) Recreate the partial-to-full JSON assembly:
   ```bash
   python Quiz-2/code/pp_combine.py
   ```
2. Score every generated JSON file using the rubric:
   ```bash
   python Quiz-2/code/goodness_score.py Quiz-2/data
   ```
   The script awards 50 points for valid JSON and 10 points for each required
   metadata field that is present (`recipe_name`, `data_provenance`,
   `macronutrients`, `ingredients`, `instructions`).
3. (Optional) Compare prompt-full outputs with the GAICO-style similarity
   helper:
   ```bash
   python Quiz-2/code/gaico_compare.py Quiz-2/data/oatmeal_pf1.json Quiz-2/data/oatmeal_pf2.json Quiz-2/data/oatmeal_pf3.json
   python Quiz-2/code/gaico_compare.py Quiz-2/data/blueberry_pf1.json Quiz-2/data/blueberry_pf2.json Quiz-2/data/blueberry_pf3.json
   ```
   This reports the Jaccard similarity of instruction/action tokens, mirroring
   the GAICO analysis performed for the optional extra credit.

## Submission checklist

- [x] Recorded three AI-vs-classical tool comparisons and summarized the energy
      delta alongside the average gap in `docs/answer-comments.md`.
- [x] Saved two original recipe transcripts in `data/original_recipe1.txt` and
      `data/original_recipe2.txt`.
- [x] Captured at least three prompt-full and one prompt-partial (combined)
      R3 JSON outputs **per recipe** — eight files total — under `data/`.
- [x] Documented optional chain-of-thought prompt experiments and stored the
      resulting JSON alongside the core outputs in `data/`.
- [x] Implemented the rubric-based scoring utility (`code/goodness_score.py`)
      and reported the results in `docs/answer-comments.md`.
- [x] Documented observations and answers to quiz questions in
      `docs/answer-comments.md` only, per instructions.
