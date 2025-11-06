# Project B: Fine-Tuning Sentiment Models on IMDB

This project now ships with reproducible code for the full workflow described in the CSCE 580 (Fall 2025) assignment brief. The implementation lives in `ProjectB/src/projectb` and the runnable entrypoints are under `ProjectB/scripts`. You will compare transformer-based and classical sentiment analysis pipelines on the [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

> **Need a walkthrough?** Follow the [step-by-step execution guide](STEP_BY_STEP.md) for exact commands that regenerate every artifact prior to submission.

> **Tip:** All training scripts default to small subsets (2k/1k examples) so they can be executed on the course VM. Increase the limits via command-line flags once you are ready for the full 25k/25k splits.

## 1. Dataset Overview
- Download the IMDB dataset (train/test splits with 25k labeled reviews each).
- Create three held-out test cases following the provided template: short (length), medium, and long reviews with different sentiment complexities.
- Document class balance and any sampling decisions (if you subsample for faster experimentation).

## 2. Preprocessing and Data Management (15 points)
- Clean raw reviews (HTML tags, contractions, etc.).
- Tokenize text for transformer models using Hugging Face tokenizers (e.g., `AutoTokenizer` for `distilbert-base-uncased`).
- For classical models, build TF–IDF and bag-of-words representations.
- Save cleaned datasets and feature matrices for reproducibility.

## 3. Baseline Transformer (50 points)
- Fine-tune `distilbert-base-uncased` twice:
  - Using Hugging Face `Trainer` (or PyTorch Lightning).
  - Using a custom PyTorch training loop similar to the Colab example.
- Log training curves, learning rates, and hyperparameters.
- Evaluate on validation/test splits and report accuracy, precision, recall, F1.
- Push the best-performing model to the Hugging Face Hub (optional but recommended).

## 4. Classical ML Comparisons (30 points)
- Train at least two baseline classifiers (e.g., logistic regression, SVM, Naive Bayes) on the same dataset representations.
- Compare metrics with fine-tuned DistilBERT models.
- Discuss differences in performance, training cost, and inference efficiency.

## 5. Analysis & Visualizations (40 points)
- **All Test Cases (30 pts):**
  - Evaluate every model on the three curated test cases; discuss sentiment confidence and misclassifications.
  - Contrast DistilBERT, fine-tuned DistilBERT, a distillation variant (e.g., GPT2 small or another transformer), and the best classical model.
  - Use charts/tables for sentence length, sentiment scores, and structure.
- **Accuracy & Loss Curves (30 pts):** Plot training/validation curves for each fine-tuned model.
- **Confusion Matrices (20 pts):** Provide matrices for DistilBERT variants and classical baselines.
- **Precision/Recall/F1 (30 pts):** Summarize metrics across all models.
- **Performance Comparison (30 pts):**
  - Measure inference latency and resource usage.
  - Discuss deployment trade-offs (GPU vs. CPU/MPS, batch sizes, quantization considerations).

## 6. Short-Answer Questions
Prepare written responses for the five reflection prompts in the brief, covering fine-tuning challenges, comparison to classical models, resource trade-offs, and deployment considerations.

## 7. Deliverables Checklist
- ✅ Data splits, curated test cases, preprocessing scripts/notebooks.
- ✅ Code for both fine-tuning runs and classical baselines (`ProjectB/scripts`).
- ✅ Report (PDF/Markdown) containing:
  - Accuracy/loss curves, confusion matrices, comparative tables.
  - Answers to evaluation questions.
- ✅ Optional: Hugging Face repo link and/or model card.

## 8. Suggested Workflow
1. **Environment Setup**
   ```bash
   cd ProjectB
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```
2. **Data Pipeline** – the utilities in `projectb.data` handle downloading IMDB, cleaning reviews, and splitting train/validation/test sets. Use them from notebooks or scripts.
3. **Baseline Modeling** – run the classical pipelines first to sanity-check the cleaned text and generate baseline metrics:
   ```bash
   python scripts/train_classical.py --limit-train 8000 --limit-test 4000
   ```
4. **Transformer Fine-Tuning (Trainer API)** – start with the managed training loop:
   ```bash
   python scripts/train_with_trainer.py --epochs 3 --limit-train 4000 --limit-test 2000
   ```
5. **Transformer Fine-Tuning (Manual Loop)** – repeat with the custom PyTorch trainer to practice lower-level control:
   ```bash
   python scripts/train_manual.py --epochs 3 --limit-train 4000 --limit-test 2000 --batch-size 12
   ```
6. **Baseline Transformers (No Fine-Tuning + GPT)** – run the additional baselines to benchmark the assignment’s “why/when to use LLMs” comparison points:
   ```bash
   python scripts/evaluate_baselines.py --limit-train 4000 --limit-test 2000
   ```
   The script emits `artifacts/base_metrics.json` for the untouched DistilBERT classifier and `artifacts/gpt_metrics.json` for the GPT-2 likelihood classifier.
7. **Evaluation & Visualization** – the scripts emit JSON summaries (see `ProjectB/artifacts/`). Load them into notebooks to plot curves, confusion matrices, and tables. You can regenerate the SVG plots with:
   ```bash
   python scripts/create_svg_plots.py --trainer artifacts/trainer_metrics.json \
       --manual artifacts/manual_metrics.json --classical artifacts/classical_metrics.json \
       --base artifacts/base_metrics.json --gpt artifacts/gpt_metrics.json
   ```
8. **Reporting** – compile figures, tables, and written analysis into the final report. Include comparisons between classical baselines, the base DistilBERT model, GPT classifier, and both DistilBERT fine-tuning strategies.

### End-to-End Rerun Helper

When you are ready for the final submission pass, you can regenerate *all* artifacts in one go with the orchestration script:

```bash
python scripts/run_pipeline.py \
    --epochs 3 \
    --batch-size 16 \
    --train-limit 25000 \
    --test-limit 25000
```

- Drop the `--train-limit/--test-limit` flags to consume the entire IMDB splits.
- Pass `--skip-baselines` if you only need the fine-tuning refresh, or `--skip-plots` to postpone SVG regeneration.
- Use `--plots-dir` to redirect the output location (defaults to `artifacts/plots`).

The helper sequentially invokes `train_classical.py`, `train_with_trainer.py`, `train_manual.py`, `evaluate_baselines.py`, and finally `create_svg_plots.py` once their JSON outputs exist, ensuring the report-ready assets stay in sync.

## 9. References
- Hugging Face DistilBERT fine-tuning tutorial (PyTorch Trainer): <https://huggingface.co/blog/sentiment-analysis-python>
- TensorFlow DistilBERT tutorial: <https://towardsdatascience.com/hugging-face-transformers-fine-tuning-distilbert-for-binary-classification-tasks-490fd192379>
- Assignment-provided notebook: <https://colab.research.google.com/drive/1B_ERSgQDLNOL8NPCvkn7s8h_vEdZ_szI>
- Test case template: <https://github.com/biplav-s/book-trustworthy-chatbot/blob/main/ai-bias/testcase-template.md>
- DistilBERT paper: <https://arxiv.org/abs/1910.01108>

## 10. Milestones & Due Date
- **Check-in:** Week of Oct 27 – confirm dataset prep and baseline results.
- **Final Submission:** Thursday, Nov 20, 2025 (report + code + data artifacts).

Document progress in a project log (e.g., `ProjectB/log.md`) noting experiments, hyperparameters, and observations.

### Repository Layout
```
ProjectB/
├── README.md                # Assignment brief + workflow
├── requirements.txt         # Minimal dependency set
├── scripts/
│   ├── evaluate_baselines.py# Base DistilBERT and GPT-2 evaluations
│   ├── train_classical.py   # TF–IDF + LR / Naive Bayes baselines
│   ├── train_manual.py      # Custom PyTorch fine-tuning loop
│   └── train_with_trainer.py# Hugging Face Trainer wrapper
├── src/projectb/
│   ├── __init__.py
│   ├── baselines.py         # Base transformer & GPT evaluations
│   ├── classical.py         # Classical ML utilities
│   ├── cleaning.py          # Text normalisation helpers
│   ├── data.py              # Dataset download/splitting/tokenisation
│   ├── manual_training.py   # Manual PyTorch loop implementation
│   ├── metrics.py           # Shared metric helpers
│   └── trainer_workflow.py  # Trainer-based workflow
└── tests/
    └── test_cleaning.py     # Example pytest unit test
```

Activate the virtual environment and run `pytest` inside `ProjectB` to execute the included unit test:

```bash
cd ProjectB
pytest
```
