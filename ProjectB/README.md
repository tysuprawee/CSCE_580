# Project B: Fine-Tuning Sentiment Models on IMDB

This project guides you through reproducing and extending the workflow described in the assignment brief for CSCE 580 (Fall 2025). You will compare transformer-based and classical sentiment analysis pipelines on the [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

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
- ✅ Code for both fine-tuning runs and classical baselines.
- ✅ Report (PDF/Markdown) containing:
  - Accuracy/loss curves, confusion matrices, comparative tables.
  - Answers to evaluation questions.
- ✅ Optional: Hugging Face repo link and/or model card.

## 8. Suggested Workflow
1. **Environment Setup:** Install `torch`, `transformers`, `datasets`, `scikit-learn`, `pandas`, `numpy`, `matplotlib/seaborn`.
2. **Data Pipeline:** Implement reusable loaders for IMDB and curated cases.
3. **Baseline Modeling:** Train classical models first to establish reference metrics.
4. **Transformer Fine-Tuning:** Use the Trainer API; then replicate with a manual loop.
5. **Evaluation & Visualization:** Automate metric computation and plotting.
6. **Reporting:** Compile figures, tables, and written analysis into the final report.

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
