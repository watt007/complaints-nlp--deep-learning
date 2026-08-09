# Use Case 1: NLP Text Classification — Complaint Category Prediction

## Objective
Predict a customer complaint's **category** (one of 6 classes) directly from its free-text description, using a classical NLP pipeline — no deep learning.

## Why this matters
Automatically routing an incoming complaint to the right category — without a human reading it first — is a real, common use case in complaint/case management systems. It speeds up triage and ensures consistent categorization across analysts.

## Pipeline

| Step | Technique | Library |
|---|---|---|
| 1. Lowercase | `.lower()` | Python |
| 2. Remove punctuation | `str.translate` | Python |
| 3. Tokenize | `word_tokenize` | NLTK |
| 4. Remove stopwords | stopword list filtering | NLTK |
| 5. Vectorize | TF-IDF (term frequency × inverse document frequency) | scikit-learn |
| 6. Split | 80/20 train/test, stratified by category | scikit-learn |
| 7. Classify | Logistic Regression | scikit-learn |
| 8. Evaluate | accuracy, precision/recall/F1 per class | scikit-learn |

## Notebook contents
`01_nlp_category_classification.ipynb` walks through every step individually — including printing the output of each preprocessing stage on a single real sentence before applying it to the full dataset, so every transformation is visible and verifiable, not just called as a black-box function.

## Results

- **Accuracy: 100%** across all 6 categories (precision/recall/F1 all 1.00)
- Verified generalization: a hand-written test sentence using different wording than any training example was still correctly classified

## Honest interpretation
This dataset's categories were designed with distinct, non-overlapping vocabulary (e.g., "unauthorized trade" complaints reliably mention stock tickers and trade terms; "fee dispute" complaints reliably mention dollar amounts and billing language). This makes the classification task cleanly separable, which is why accuracy is near-perfect. Real-world complaint data would likely show more vocabulary overlap between categories, and true production accuracy would be expected to be meaningfully lower than 100%. This result demonstrates the pipeline is built and functioning correctly — not that the general problem of complaint classification is "solved."

## How to run
```bash
conda create -n complaints_nlp python=3.11
conda activate complaints_nlp
conda install jupyter pandas scikit-learn nltk -y
jupyter notebook 01_nlp_category_classification.ipynb
```
