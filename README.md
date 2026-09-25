# 🛡️ NewsDetective AI: Misinformation & Content Reliability Detection System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-99.37%25-brightgreen.svg)]()

> An intelligent, end-to-end Machine Learning and Natural Language Processing platform that evaluates news articles and user-submitted text to determine content reliability, detect sensationalist clickbait, and provide transparent feature-level explanations for its predictions.

---

## 📌 Problem Statement

In today's digital era, misleading information, fabricated claims, and emotionally charged clickbait proliferate rapidly across social networks and news platforms. Naive detection models often suffer from **dataset leakage** (e.g., memorizing news agency wire prefixes like `Reuters` instead of actual linguistic cues) or operate as **opaque black boxes** without giving readers tangible reasons why an article was flagged.

### Core Objective:
> *"Build an AI/ML system that analyzes a news article or user-provided text and predicts whether the content is potentially reliable or misleading. The system should analyze textual patterns and provide an explanation or confidence score for its prediction."*

### Key Requirements Delivered:
- 📰 **News/Article Text Input**: Flexible input through manual text/headline entry, text/CSV file upload, and pre-loaded benchmark presets.
- 🧹 **Text Preprocessing**: Robust agency dateline stripping, HTML/URL sanitization, punctuation normalization, and token-level featurization.
- 🤖 **ML Classification**: Rigorous multi-model comparison across **44,898 articles** (ISOT benchmark dataset).
- 📈 **Confidence Score**: Calibrated probability distributions across 5 reliability tiers with uncertainty thresholds.
- 💡 **Key Indicators & Reasoning**: Explainable AI (XAI) featuring token log-odds feature attribution ($w_i \cdot x_i$), stylometric sensationalism analysis, and in-context colored text highlighting.
- 🖥️ **Result Dashboard**: High-performance, modern Streamlit web dashboard with interactive Plotly visual analytics and batch processing.

---

## 🏗️ System Architecture

The NewsDetective AI pipeline comprises four decoupled layers engineered for speed, scientific rigor, and explainability:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. INGESTION LAYER                              │
│   • Raw Headline & Body Text   • Benchmark Presets   • Batch CSV Upload │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  2. PREPROCESSING & STYLOMETRICS                       │
│   • Dateline Removal: Strips wire prefixes ("WASHINGTON (Reuters) -")   │
│   • Sanitization: Removes HTML, URLs, bracketed artifacts, numbers     │
│   • Sensationalism Diagnostics: ALL-CAPS ratio, '!' & '?' frequencies  │
│   • Linguistic Profiling: Polarity, Subjectivity, Flesch Reading Ease  │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│       3A. TF-IDF VECTORIZATION       │  │ 3B. STYLOMETRIC METRICS      │
│   • Sublinear Term Frequency         │  │ • Sensationalism Index (0-100│
│   • Unigrams & Bigrams (1, 2)        │  │ • Clickbait Trigger Matches  │
│   • 25,000 N-Gram Feature Dimensions │  │ • Subjectivity Score         │
└───────────────────┬──────────────────┘  └──────────────┬───────────────┘
                    │                                    │
                    ▼                                    │
┌──────────────────────────────────────┐                 │
│      4. MACHINE LEARNING ENGINE      │                 │
│   • Logistic Regression (Primary)    │                 │
│   • Passive-Aggressive Classifier    │                 │
│   • Multinomial Naive Bayes          │                 │
│   • Random Forest Classifier         │                 │
└───────────────────┬──────────────────┘                 │
                    │                                    │
                    ▼                                    │
┌────────────────────────────────────────────────────────┴───────────────┐
│               5. EXPLAINABILITY & REASONING (XAI)                      │
│   • Linear Feature Attribution: Score_i = TFIDF_i × Weight_i           │
│   • In-Context HTML Highlighter: Green (Reliable) vs Red (Deceptive)   │
│   • Calibrated Confidence Score & 5-Tier Classification                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  6. PRESENTATION & INTERACTION                         │
│   • Live Streamlit Web Dashboard (http://localhost:8501)               │
│   • Command-Line Interface (predict.py) & JSON API                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ / 3.14 | Core programming language |
| **Machine Learning** | `scikit-learn` 1.8.0 | Classification algorithms, TF-IDF vectorization, metrics, probability calibration |
| **Serialization** | `joblib` | High-compression model artifact persistence |
| **NLP & Text Processing** | `re`, `string`, `textblob`, `nltk` | Regex-based dateline cleaner, sentiment polarity, and subjectivity scoring |
| **Data Manipulation** | `pandas` 2.3+, `numpy` 2.3+ | Tabular data processing and numerical transformations |
| **Web Dashboard** | `streamlit` 1.55.0 | Interactive, real-time UI dashboard |
| **Data Visualization** | `plotly` 6.6.0 | Interactive feature importance charts, pie charts, and confusion matrix heatmaps |
| **Testing** | `unittest` | Unit and integration test suite |

---

## 📊 Model Performance & Benchmarks

The models were trained and tested on the **ISOT Fake News Dataset** comprising **44,898 total samples** (23,481 Fake, 21,417 True) with an **80/20 stratified train-test split**:

| Model | Accuracy | Precision | Recall | F1-Score | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Primary)** | **99.37%** | **99.28%** | **99.40%** | **99.34%** | **< 10ms** |
| **Passive-Aggressive Classifier** | **99.68%** | **99.63%** | **99.70%** | **99.66%** | **< 15ms** |
| **Random Forest Classifier** | 99.30% | 99.44% | 99.10% | 99.27% | ~ 35ms |
| **Multinomial Naive Bayes** | 96.39% | 96.15% | 96.30% | 96.22% | < 5ms |

> **Note on Feature Attribution**: Logistic Regression is utilized as the primary explanation engine because its linear decision boundary allows direct, uncorrupted feature attribution ($w_i \cdot x_i$), enabling users to clearly see which exact words influenced the prediction.

---

## 🚀 Key Features

### 1. 📡 Live Breaking News Stream & Real-Time Predictions
- Streams live headlines directly from global news wire services across multiple categories (Top Stories, Politics, Technology, Business, Science, Health, World).
- Features on-demand topic/keyword search (e.g., query "Trump", "AI", "climate", "crypto").
- Automatically evaluates each incoming live story in real time with prediction badges (`RELIABLE` vs `MISLEADING`), confidence scores, sensationalism ratings, and publisher attribution.

### 2. In-Context Word Highlighting & Diagnostics
- Visualizes word-by-word influence directly within the original text.
- **Green highlights**: High-credibility journalistic markers (e.g., *spokesman, budget, confirmed, treaty, official, agency*).
- **Red highlights**: Tabloid markers and sensationalism (e.g., *bombshell, deep state, exposed, panic, mainstream media*).

### 2. Multi-Model Consensus
- Displays side-by-side probability estimates from Logistic Regression, Passive-Aggressive, and Naive Bayes to ensure consistency.

### 3. Sensationalism & Stylometric Index (0–100)
- Analyzes uppercase shouting ratios, exclamation mark density, subjective sentiment, and provocative trigger vocabulary.

### 4. Batch CSV Scanning
- Upload datasets with hundreds of articles, generate instantaneous predictions, view aggregate distribution charts, and export results with one click.

---

## ⚡ Quick Start Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/rk2725q-star/ai-fake-news-detect.git
cd ai-fake-news-detect
```

Ensure dependencies are installed:
```bash
pip install pandas numpy scikit-learn streamlit plotly textblob joblib
```

### 2. Launch the Interactive Dashboard

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 3. Run Standalone CLI Predictions

```bash
# Analyze direct text
python predict.py "BOMBSHELL: Secret Deep-State Microchip Program Exposed By Whistleblower! Wake up people!"

# Analyze from file
python predict.py --file path/to/article.txt

# JSON Output Mode (for API integrations)
python predict.py --json "Global investment in renewable energy reached a record high..."
```

### 4. Run the Test Suite

```bash
python -m unittest tests/test_detector.py
```

---

## 📁 Repository Structure

```
├── Datasets/
│   ├── datasets.zip                  # Raw zipped dataset archive
│   ├── Fake.csv                      # ISOT fake news corpus (23,481 records)
│   ├── True.csv                      # ISOT genuine news corpus (21,417 records)
│   └── manual_testing.csv            # Manual verification test set
├── models/
│   ├── fake_news_detector.joblib     # Serialized vectorizer & multi-model bundle
│   └── metrics.json                  # Performance benchmark metrics & vocabulary logs
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Dateline stripping, NLP cleaning & stylometrics
│   ├── model_trainer.py              # Stratified training pipeline & model comparison
│   └── explainer.py                  # XAI feature attribution, highlighting & consensus
├── tests/
│   └── test_detector.py              # Automated unit and integration test suite
├── app.py                            # Streamlit web application & result dashboard
├── predict.py                        # Standalone CLI tool and JSON inference engine
├── sample_articles.py                # Pre-loaded benchmark testing presets
└── README.md                         # Project documentation
```

---

## 👥 Team Details & Contributors

| Role | Name | Responsibilities / Contribution |
| :--- | :--- | :--- |
| **Lead Developer / AI Engineer** | RANJITHKUMAR (@rk2725q-star) | Full System Architecture, ML Model Pipeline, XAI & Streamlit UI |
| **Collaborator** | **GOVINDHARAJ S** | Dataset Benchmarking & Analysis |
| **Collaborator** | **KISHORE R** | Testing & Quality Assurance |


---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
