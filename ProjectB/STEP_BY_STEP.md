# Project B Step-by-Step Guide

This guide explains how to reproduce every deliverable required by the CSCE 580 Project B assignment using the tooling included in this repository.

> **Prerequisites**
> * Linux, macOS, or WSL environment with Python 3.10+
> * At least 16 GB RAM and a GPU (optional but recommended for full-dataset fine-tuning)
> * Active internet connection to download the IMDB dataset and Hugging Face models
> * A Hugging Face access token if you plan to push trained models to the Hub

---

## 1. Clone and Set Up the Environment

```bash
cd ProjectB
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Troubleshooting tips:

* If PyTorch fails to install with GPU support, reinstall using the platform-specific command from <https://pytorch.org/get-started/locally/>.
* When running on CPU-only hardware, expect longer training times (45–90 minutes per epoch on the full dataset).

---

## 2. Download and Inspect the Data

The scripts automatically download the IMDB Movie Review dataset using `datasets.load_dataset("imdb")`. To verify the download beforehand, run:

```bash
python -c "from datasets import load_dataset; ds = load_dataset('imdb'); print(ds)"
```

Optional: Save a local copy for offline use by passing `--cache-dir <path>` to any script.

---

## 3. Generate Baseline Features (Classical Models)

Train the TF–IDF-based classical classifiers. Adjust `--limit-train` and `--limit-test` if you want to practice on smaller subsets before the full 25k/25k splits.

```bash
python scripts/train_classical.py --limit-train 25000 --limit-test 25000 --output artifacts/classical_metrics.json
```

Artifacts produced:

* `artifacts/classical_metrics.json` – evaluation metrics, confusion matrix, run metadata
* `artifacts/tfidf_vectorizer.joblib` – serialized TF–IDF vocabulary (for reuse in notebooks)

---

## 4. Fine-Tune DistilBERT with Hugging Face Trainer

```bash
python scripts/train_with_trainer.py \
    --epochs 3 \
    --train-batch-size 16 \
    --eval-batch-size 32 \
    --limit-train 25000 \
    --limit-test 25000 \
    --output artifacts/trainer_metrics.json
```

Both the hyphenated flags shown above and the underscore versions (for example,
`--train_batch_size`) are accepted, so feel free to use whichever style matches
your shell history or personal notes.

Artifacts produced:

* `artifacts/trainer_metrics.json` – per-epoch loss/accuracy plus evaluation metrics
* `artifacts/trainer_model/` – best model checkpoint (safe to push to the Hub)
* `artifacts/trainer_confusion.npy` – confusion matrix saved as a NumPy array

**GPU tip:** Add `--no-cuda` to force CPU execution; remove `--limit-*` flags to use the entire dataset.

---

## 5. Fine-Tune DistilBERT with the Custom PyTorch Loop

```bash
python scripts/train_manual.py \
    --epochs 3 \
    --batch-size 12 \
    --learning-rate 2e-5 \
    --limit-train 25000 \
    --limit-test 25000 \
    --output artifacts/manual_metrics.json
```

Artifacts produced:

* `artifacts/manual_metrics.json` – per-epoch history and evaluation scores
* `artifacts/manual_state_dict.pt` – PyTorch state dict for the fine-tuned model

**Checkpoints:** Use `--save-every` to enable periodic checkpointing in long runs.

---

## 6. Evaluate Base DistilBERT and GPT-2 Baselines

```bash
python scripts/evaluate_baselines.py --limit-test 25000 --output-dir artifacts
```

This command writes:

* `artifacts/base_metrics.json` – results for the zero-shot DistilBERT classifier (`distilbert-base-uncased-finetuned-sst-2-english`)
* `artifacts/gpt_metrics.json` – GPT-2 likelihood baseline, including latency statistics

Use `--testcases ProjectB/data/testcases.json` to evaluate the curated GAICO examples simultaneously.

---

## 7. Regenerate Plots and Tables

Once all JSON files exist, create SVG plots for loss/accuracy curves and confusion matrices:

```bash
python scripts/create_svg_plots.py \
    --trainer artifacts/trainer_metrics.json \
    --manual artifacts/manual_metrics.json \
    --classical artifacts/classical_metrics.json \
    --base artifacts/base_metrics.json \
    --gpt artifacts/gpt_metrics.json \
    --output-dir artifacts/plots
```

Outputs include:

* `trainer_accuracy.svg`, `trainer_loss.svg`, `manual_*` – learning curves
* `*_confusion.svg` – confusion matrices for each model family

These SVGs are ready to embed in the final report (convert to PNG/PDF if needed).

---

## 8. Run Everything at Once (Optional)

Skip the manual steps above by executing the orchestration helper. Use conservative `--train-limit` values if you want a smoke test before the full run.

```bash
python scripts/run_pipeline.py --epochs 3 --batch-size 16 --train-limit 25000 --test-limit 25000
```

Flags:

* `--skip-classical`, `--skip-trainer`, `--skip-manual`, `--skip-baselines`, `--skip-plots` – bypass specific stages
* `--plots-dir <path>` – redirect generated SVGs
* `--cache-dir <path>` – reuse datasets between runs

The script stops on the first failing stage and prints the command it attempted so you can rerun it manually.

---

## 9. Assemble the Report

1. Copy `ProjectB/report.md` into your preferred writing tool or export it directly to PDF (e.g., `pandoc report.md -o report.pdf`).
2. Replace the placeholder metric tables with the values from your latest JSON artifacts.
3. Embed the confusion matrices and learning-curve SVGs.
4. Update the answers to the five reflection questions with insights from your final runs.
5. Include a summary of time-to-train and inference latency from each JSON file's `timing` section.

---

## 10. Package Deliverables

Before submission, ensure the following files are up to date:

* `ProjectB/artifacts/*.json` – metric summaries for all models
* `ProjectB/artifacts/plots/*.svg` – visualizations referenced in the report
* `ProjectB/data/testcases.json` – curated GAICO test cases with outcomes
* `ProjectB/report.md` (and PDF export) – final report
* `ProjectB/scripts/` and `ProjectB/src/projectb/` – source code used to produce results

Run `pytest` to confirm unit tests pass:

```bash
cd ProjectB
pytest
```

Finally, create an archive for submission:

```bash
cd ..
zip -r ProjectB_submission.zip ProjectB -x "*.venv*" "__pycache__/*"
```

Good luck with your experiments! Reach out to the course staff if you encounter hardware or dependency issues.
