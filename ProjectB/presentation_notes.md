# Project B: Presentation Script & Insights

## Slide Script: "Regular Project - Suprawee Pongpeeradech"

**Speaker Notes:**

"Hi everyone, I'm Suprawee, and I'd like to share the results of my sentiment analysis project on the IMDB dataset.

**Status & Results:**
First, looking at the status, I'm happy to report that the fine-tuned DistilBERT model achieved a **91.4% accuracy** on the test set. This successfully outperformed our baselines: the classical Logistic Regression model came in close at **89.4%**, while the zero-shot GPT-2 approach lagged behind at **78.0%**. I've generated and verified all the necessary artifacts, including the metrics and plots you see in the report.

**Key Experiences & Learnings:**
Moving on to my experience, this project really highlighted the trade-offs between modern transformers and classical methods.
- **Transformers vs. Classical:** I learned that while classical models like TF-IDF with Logistic Regression are incredibly fast and surprisingly effective—reaching nearly 90% accuracy—they hit a ceiling. Fine-tuned transformers like DistilBERT provide that crucial edge in capturing nuance and context to push us over the 90% barrier.
- **The Importance of Fine-tuning:** A major takeaway was observing the base DistilBERT model. Without fine-tuning, it was essentially guessing, with **49% accuracy**. This really drove home the critical need for task-specific supervision; you can't just take a raw masked-language model and expect it to classify sentiment out of the box.
- **Trade-offs:** Finally, I gained a lot of experience balancing performance versus cost. DistilBERT gives the best results but is much slower to train. GPT-2 offers a cool zero-shot capability, but it's computationally expensive for inference.

**Demonstration:**
On the right, you can see the confusion matrix for our best model. It shows a healthy balance between true negatives and true positives, though like many models, it still occasionally struggles with sarcastic or mixed-sentiment reviews.

Overall, this project demonstrated that while classical methods are great for a quick start, fine-tuning transformers is essential for state-of-the-art performance."

---

## Project Insights

### What I Learned
1.  **Transformers vs. Classical Approaches:**
    *   **Classical (TF-IDF + Logistic Regression):** I was surprised by how effective simple linear models are. They are extremely fast to train (seconds/minutes) and inference is negligible (<1ms). They achieved ~89% accuracy, which is sufficient for many use cases.
    *   **Transformers (DistilBERT):** They offer superior performance (91.4%) by understanding context (e.g., "not bad" is positive, whereas bag-of-words might see "bad" and think negative). However, the computational cost is significantly higher.

2.  **The Necessity of Fine-Tuning:**
    *   I observed that the `distilbert-base-uncased` model has no inherent notion of "positive" or "negative" sentiment. It yielded ~49% accuracy (random chance) before fine-tuning. This taught me that pre-trained models capture language structure, but task-specific heads and fine-tuning are mandatory for classification tasks.

3.  **Implementation Frameworks:**
    *   I learned to use the **Hugging Face Trainer API** for rapid development and standardized loops.
    *   I also implemented a **Custom PyTorch Loop**, which gave me deeper insight into the training process (gradient clipping, scheduler stepping, batch management), even though it was more code to maintain.

### Limitations
1.  **Computational Cost:**
    *   **Training:** Fine-tuning DistilBERT took significantly longer (~30 minutes on CPU) compared to Logistic Regression (seconds).
    *   **Inference:** Real-time inference with transformers (~4-5ms per sample) is orders of magnitude slower than classical models. GPT-2 likelihood scoring was even slower (~20ms) due to token-by-token processing.

2.  **Data Requirements:**
    *   The high performance of DistilBERT relies heavily on the availability of the 25,000 labeled training examples. The zero-shot GPT-2 approach (which doesn't need training data) performed significantly worse (78%), showing that we still need labeled data for top-tier results.

3.  **Context Window:**
    *   DistilBERT has a maximum sequence length (typically 512 tokens). Longer reviews might get truncated, potentially losing critical sentiment information at the end of the text.

### Encountered Problems & Solutions
1.  **Base Model "Failure":**
    *   *Problem:* Initially, I might have expected the pre-trained DistilBERT to work "out of the box". It returned ~49% accuracy.
    *   *Solution:* I realized this was expected behavior for a masked language model. The solution was adding a classification head and fine-tuning it on the IMDB dataset.

2.  **Manual Loop Instability:**
    *   *Problem:* The custom PyTorch training loop was more sensitive to hyperparameters (like batch size and learning rate) than the Trainer API, which has good defaults.
    *   *Solution:* I had to ensure proper gradient clipping and learning rate scheduling were implemented to stabilize the training loss.

3.  **Ambiguous Reviews:**
    *   *Problem:* The model struggled with "mixed" reviews (e.g., "The acting was great, but the plot was terrible").
    *   *Analysis:* As seen in the `long_mixed` test case, models can get confused by conflicting signals. While DistilBERT handles this better than bag-of-words, it's still a challenge.
