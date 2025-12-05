# Mockup Project: Multi-modal AI Content Detector

## Slide Script: "Multi-modal AI Content Detector - SP"

**Speaker Notes:**

"For my proposed future project, I'd like to introduce the **Multi-modal AI Content Detector**.

**Project Context & Goal:**
The core idea here is to tackle the rising threat of misinformation and deepfakes. We can't just look at text or images in isolation anymore. My proposed system analyzes **text, images, and videos** together to detect AI-generated content.
This tool is designed for the people on the front lines: **social media platforms, journalists, fact-checkers, and news organizations**.

**Technical Approach:**
To achieve this, I'm proposing a robust ensemble of models:
-   For **Text**, I'll use a fine-tuned **RoBERTa** model.
-   For **Images**, an **EfficientNet-B7**.
-   For **Videos**, a **3D-CNN** to capture temporal inconsistencies.
-   These signals will be combined using an **XGBoost ensemble** to make the final decision.

**Data & Evaluation:**
I plan to train this on a massive dataset: 500k text samples, 200k synthetic images, and 33k deepfake videos.
Preliminary evaluation targets are promising: we're aiming for **92% AUC-ROC for text** and **88% for images**, with a real-time analysis speed of **under 2 seconds**.

**Mockup Demonstration:**
Let's look at the mockup on the right to see how it works in practice.
-   **Input:** A user submits a social media post containing a breaking news image and a caption.
-   **Analysis:** The system scans both components.
-   **Output:** It flags the image as **87% AI-generated**, triggering a **Red Warning**. Meanwhile, the text is analyzed as 23% likely to be AI (which is safe).
-   **Verdict:** The system assigns an overall **'Medium Risk'** assessment. This allows a human moderator to prioritize this post for review before it spreads.

**Challenges:**
We are aware of the trust issues involved—specifically the risk of false positives harming legitimate content and the constant 'arms race' against improving generative models. But this multi-modal approach is our best defense."
