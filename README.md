# 🧠 Complaints Intelligence — NLP & Deep Learning Portfolio Project

**Two applied ML use cases built on a single synthetic customer complaints dataset** — a financial services domain project demonstrating classical NLP, deep learning with TensorFlow/Keras, and honest model evaluation.

> **Note:** All data in this project is synthetically generated for portfolio/demonstration purposes. No real customer or institutional data is used.

---

## 📌 Project Synopsis

Financial institutions handling brokerage and non-brokerage customer complaints need to understand and route complaints efficiently — classifying what a complaint is about, and predicting which complaints are likely to escalate, both directly from the free-text a customer writes. This project builds and compares two independent machine learning approaches to that problem on the same underlying dataset:

- **Use Case 1 — NLP Text Classification**: predict the complaint **category** (6 classes) from the raw complaint description, using classical NLP techniques.
- **Use Case 2 — Deep Learning**: predict complaint **escalation risk** (binary) from the same kind of text, using a neural network built with TensorFlow/Keras.

The project intentionally builds both a simple baseline and a deep learning model on comparable problems, so the results can be **honestly compared** — a core, real-world data science skill that goes beyond "I trained a model."

## 🎯 Objectives

1. Demonstrate an end-to-end classical NLP pipeline: text cleaning, tokenization, stopword removal, TF-IDF vectorization, and classification.
2. Demonstrate an end-to-end deep learning NLP pipeline: Keras tokenization, sequence padding, an Embedding layer, and a trained neural network.
3. Evaluate both approaches rigorously (train/test split, accuracy, precision/recall/F1, generalization tests on unseen sentences).
4. Draw an honest, evidence-based conclusion about when deep learning does — and does not — outperform a simpler baseline.

---

## 📊 The Dataset

`complaints_ml_dataset.csv` — **6,000 rows, 18 fields**, synthetically generated with genuine, verifiable signal (not random).

| Field | Description |
|---|---|
| `case_id` | Unique complaint case identifier |
| `account_number` | Masked account number |
| `account_type` | Brokerage / Non-Brokerage |
| `customer_name` | Synthetic customer name |
| `customer_since_year` | Year the customer's account was opened |
| `broker_assigned` | Broker/advisor ID assigned to the case |
| `complaint_type` | Service / Escalated |
| `complaint_category` | One of 6 categories (see below) |
| `complaint_description` | Free-text customer complaint — **primary NLP input** |
| `resolution_notes` | Free-text resolution summary |
| `channel` | Phone / Secure Message / Branch / Email / Mobile App |
| `severity` | Low / Medium / High |
| `status` | Closed / Pending Review / In Progress / Escalated to L2 |
| `complaint_date`, `resolution_date`, `resolution_days` | Case timing fields |
| `customer_satisfaction_score` | 1–5, where applicable |
| `escalated_flag` | 0/1 — **primary Use Case 2 target** |

**Complaint categories:** Unauthorized Trade, Fee Dispute, Processing Delay, Account Access Issue, Statement Error, Overdraft/NSF Dispute — each with distinct, category-specific vocabulary by design, so classification results are interpretable and explainable.

---

## 🔤 Use Case 1: NLP Text Classification

**Goal:** predict `complaint_category` from `complaint_description` alone.

### Pipeline
```
Raw text
  → lowercase
  → remove punctuation
  → tokenize (NLTK word_tokenize)
  → remove stopwords (NLTK stopword list)
  → TF-IDF vectorization (scikit-learn TfidfVectorizer)
  → train/test split (80/20, stratified)
  → Logistic Regression classifier
  → evaluation
```

### Key techniques demonstrated
- Manual, step-by-step text preprocessing (each transformation stage inspected individually before automating)
- TF-IDF: term-frequency × inverse-document-frequency vectorization, explained and verified by hand
- Stratified train/test splitting to preserve class balance
- `classification_report` (precision/recall/F1 per class)
- Generalization testing on hand-written sentences never seen in training

### Result
- **Accuracy: ~100%** (1.00 precision/recall/F1 across all 6 categories)
- **Why so high:** the dataset's category-specific vocabulary is cleanly separable by design — a valuable, honest caveat for this result. Real-world complaint text is messier (overlapping vocabulary, ambiguity, typos), so production accuracy would be expected to be lower. This result validates that the *pipeline* works correctly, not that the *problem* is solved in general.
- Correctly classified new, unseen sentences with different phrasing than any training example (e.g., a wire-transfer-delay sentence worded differently from the training templates was still correctly classified as "Processing Delay").

---

## 🧠 Use Case 2: Deep Learning (TensorFlow/Keras)

**Goal:** predict `escalated_flag` (will this complaint be escalated?) from `complaint_description` alone, using a neural network.

### Pipeline
```
Raw text
  → same NLTK cleaning as Use Case 1
  → Keras Tokenizer: word → integer ID mapping (vocab size 2000, OOV token handling)
  → pad_sequences: every sentence forced to a fixed length (40 tokens)
  → train/test split (80/20, stratified)
  → Neural network (see architecture below)
  → evaluation + generalization test
```

### Model architecture
```
Input(shape=(40,))
  → Embedding(vocab_size=2000, embedding_dim=16)      # 32,000 params — learned word representations
  → GlobalAveragePooling1D()                           # 0 params — averages word embeddings into 1 sentence vector
  → Dense(16, activation="relu")                        # 272 params — learned hidden layer
  → Dense(1, activation="sigmoid")                       # 17 params — outputs escalation probability (0–1)

Total trainable parameters: 32,289 (126.13 KB)
```

### Key techniques demonstrated
- Keras `Tokenizer` and `pad_sequences` — the sequence-based alternative to TF-IDF, which preserves word order (unlike TF-IDF)
- A trainable `Embedding` layer — dense, learned word representations vs. TF-IDF's fixed, hand-designed scoring
- A complete `Sequential` Keras model: built, compiled (`adam` optimizer, `binary_crossentropy` loss), trained for 15 epochs
- Train/validation accuracy tracked per epoch to monitor overfitting
- Generalization testing on a hand-written, never-seen sentence

### Result
- **Training accuracy: 69.3% | Validation accuracy: 67.08%** (close together — no overfitting)
- Training plateaued after epoch 6, indicating the model reached the limit of exploitable signal for this dataset size
- **Comparable to the TF-IDF baseline (~67%)** tested earlier on the same escalation-prediction problem

### Honest conclusion — the most important finding of this project
**The deep learning model did not outperform the simpler TF-IDF baseline on this dataset.** This is a real, common, and important finding in applied machine learning: deep learning's advantages typically emerge on **larger, messier, more linguistically complex datasets** where it can learn subtleties a simpler model cannot. On a smaller, cleaner dataset like this one, a simpler model can perform just as well with far less computational cost — a genuine engineering trade-off, not a failure of the deep learning approach. Recognizing *when* to reach for deep learning vs. a simpler baseline is itself a core data science skill this project demonstrates.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Classical NLP | NLTK (tokenization, stopwords), scikit-learn (TF-IDF, Logistic Regression, train/test split, metrics) |
| Deep Learning | TensorFlow / Keras (Tokenizer, Embedding, Dense layers, model training) |
| Data handling | pandas |
| Environment | Conda (isolated environment per project stage) |

## 📁 Repository Structure

```
complaints-ai-project/
├── README.md                              # this file
├── data/
│   └── complaints_ml_dataset.csv          # synthetic dataset (6,000 rows, 18 fields)
├── use_case_1_nlp_classification/
│   ├── 01_nlp_category_classification.ipynb
│   └── README.md
├── use_case_2_deep_learning/
│   ├── 02_escalation_prediction_dl.ipynb
│   └── README.md
├── requirements.txt
└── docs/
    └── screenshots/                       # model summary, training curves, etc.
```

## ▶️ Running Locally

```bash
git clone <your-repo-url>
cd complaints-ai-project

# Use Case 1 environment
conda create -n complaints_nlp python=3.11
conda activate complaints_nlp
conda install jupyter pandas scikit-learn nltk -y
jupyter notebook use_case_1_nlp_classification/01_nlp_category_classification.ipynb

# Use Case 2 environment (separate, since TensorFlow is heavier)
conda create -n dl_env python=3.10
conda activate dl_env
conda install jupyter pandas scikit-learn nltk -y
pip install tensorflow-cpu
jupyter notebook use_case_2_deep_learning/02_escalation_prediction_dl.ipynb
```

## 📈 Possible Extensions

- Scale the dataset up significantly (10x–100x) to test whether deep learning's advantage over TF-IDF emerges at larger scale, as theory predicts
- Add CNN (Conv1D) and RNN (LSTM/GRU) architecture variants to Use Case 2 and compare all three deep learning approaches head-to-head
- Swap the small custom Embedding layer for pretrained embeddings (e.g., GloVe) to test whether transfer learning improves results on this dataset size
- Deploy Use Case 1 as a live demo (Streamlit) for interactive category classification

## 👤 About This Project

Built as a hands-on demonstration of applied NLP and deep learning, using a financial-services complaints domain consistent with real analytics/data engineering experience in banking and conduct-risk analytics. Emphasis throughout on understanding every mechanism by hand (manual preprocessing walkthroughs, parameter-count verification, honest evaluation) rather than treating model training as a black box.
