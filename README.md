# 🎬 CineMetrics: Cinema Intelligence & Sentiment Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![NLTK](https://img.shields.io/badge/NLTK-VADER-brightgreen.svg)](https://www.nltk.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM%20%7C%20K--Means-orange.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-ff6384.svg?logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)

An advanced, end-to-end cinema intelligence and sentiment analytics web application. CineMetrics solves common limitations of off-the-shelf NLP sentiment tools—specifically misclassifying horror/thriller praise, missing subtle sarcasm, and lacking entity-level movie intelligence.

---

## ✨ Key Features

1. **🎭 8-Genre Semantic Classification:**
   - Automatically detects movie genre (`Sci-Fi`, `Action`, `Horror`, `Thriller`, `Comedy`, `Romance`, `Drama`, `Animation`) from review text using weighted keyword matching (3x weight for explicit genre tokens) with normalized confidence percentages and extracted clue tokens.

2. **👻 Cinema Domain-Calibrated Sentiment Engine:**
   - Standard NLP models wrongly score horror words like *"terrifying"*, *"scary"*, and *"chilling"* as negative. CineMetrics recalibrates these to positive praise while guarding against true criticisms (*"not scary"*, *"waste of time"*, *"boring"*).

3. **🙃 Sarcasm & Irony Inversion:**
   - Regex-based syntactic pattern matcher that catches ironic starters (*"Oh great"*, *"Yeah, because"*), conditional irony (*"masterpiece... if you have no brain"*), and scare quotes, inverting polarity to `Negative (Sarcastic) 🙃`.

4. **🔍 Movie ID Intelligence Hub:**
   - Enter any Movie ID (e.g. `M2024_01`) to view aggregated metrics: average star rating, box office gross ($ millions), dominant sentiment consensus bar, and complete review list.

5. **⚡ Bi-Directional Film Archive Filtering:**
   - 200-review archive table (50 movies, 2015–2024) located directly below search, featuring instant client-side DOM filtering by Movie ID and genre with a one-click reset button.

6. **📊 10-Year Longitudinal Trend Visualizations:**
   - Interactive Chart.js graphs displaying 10-year ratings vs. box office trends and genre sentiment polar area distributions.

---

## 🧠 Machine Learning Architecture

The platform integrates two core Machine Learning models evaluated **strictly on Precision, Recall, and F1-Score**:

```
                   ┌──────────────────────────────────────────────┐
                   │          Machine Learning Algorithms         │
                   └──────────────────────┬───────────────────────┘
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 ▼                                                 ▼
  ┌─────────────────────────────┐                   ┌─────────────────────────────┐
  │   1. Supervised Learning    │                   │   2. Unsupervised Learning  │
  │ Support Vector Machine(SVM) │                   │      K-Means Clustering     │
  └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                 │                                                 │
  • Uses labeled review training data               • Operates without sentiment labels
  • Finds optimal separating hyperplane             • Groups reviews into K=3 clusters
  • Classifies: Positive, Neutral, Negative         • Discovers natural text groupings
```

### 1. Supervised Learning: Support Vector Machine (SVM)
- **Model:** `LinearSVC` with TF-IDF Vectorization (500 features).
- **Function:** Trains on labeled reviews to learn the optimal separating decision boundary (hyperplane) across Positive, Neutral, and Negative classes.

### 2. Unsupervised Learning: K-Means Clustering
- **Model:** `KMeans(n_clusters=3)`.
- **Function:** Clusters unlabeled review embeddings based on centroid distance, automatically grouping praise vs. critique.

---

## 🎯 Evaluation Parameters & Performance

Evaluation parameters are strictly focused on:
- **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (Accuracy of predicted classes)
- **Recall:** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (Coverage of actual classes)
- **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ (Harmonic balance)

### 📊 Benchmark Results on `dataset.csv` (200 Reviews)

| Model | Class | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Supervised SVM** | **Positive** | **1.00** | **1.00** | **1.00** |
| | **Neutral** | **1.00** | **1.00** | **1.00** |
| | **Negative** | **1.00** | **1.00** | **1.00** |
| | **Overall (Macro)** | **1.00** | **1.00** | **1.00** |
| **Unsupervised K-Means** | **Positive** | **0.70** | **1.00** | **0.82** |
| | **Neutral** | **1.00** | **0.43** | **0.60** |
| | **Negative** | **0.00** | **0.00** | **0.00** |
| | **Overall (Macro)** | **0.57** | **0.48** | **0.47** |
| **Domain-Adapted VADER** | **Positive** | **0.98** | **0.97** | **0.98** |
| | **Neutral** | **0.78** | **0.82** | **0.80** |
| | **Negative** | **0.92** | **0.92** | **0.92** |
| | **Overall (Macro)** | **0.89** | **0.90** | **0.90** |

---

## 📁 Repository Structure

```text
sentiment_analysis_app/
│
├── app.py                                         # Flask server, routes, domain VADER, sarcasm & genre engine
├── ml_evaluation.py                               # Standalone SVM & K-Means evaluation script
├── dataset.csv                                    # 200 film reviews across 50 movies (2015–2024)
├── requirements.txt                               # Project dependencies (Flask, nltk, scikit-learn)
├── .gitignore                                     # Standard Python gitignore
│
├── templates/
│   └── index.html                                 # Reactive UI (Search, Movie Hub, Archive Table, Charts)
│
├── static/
│   └── style.css                                  # Dark Cinema theme, glassmorphism, animations
│
├── CODE_EXPLANATION_AND_EXECUTION_GUIDE.md        # Document 1: Line-by-line code explanation & manual
└── PROJECT_REPORT_ARCHITECTURE_AND_EVALUATION.md  # Document 2: Architecture report, flowcharts & metrics
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/sentiment-analysis-app.git
cd sentiment-analysis-app
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

### 5. Run the Machine Learning Evaluation
To view the Precision, Recall, and F1-Score for SVM and K-Means:
```bash
python ml_evaluation.py
```

---

## 📡 REST API Reference

### 1. Predict Sentiment / Genre or Movie ID
- **Endpoint:** `POST /predict`
- **Payload:** `{"review_text": "Terrifying horror with incredible jump scares!"}`
- **Response:**
  ```json
  {
    "badge_label": "Positive",
    "color": "success",
    "compound": 0.8912,
    "emoji": "😊",
    "genre_clues": ["horror", "jump scares", "terrifying"],
    "genre_confidence": "94%",
    "genre_detected": "Horror",
    "genre_emoji": "👻",
    "is_movie_id": false,
    "is_sarcastic": false,
    "sentiment": "Positive"
  }
  ```

### 2. Fetch Movie Intelligence by ID
- **Endpoint:** `GET /movie/<movie_id>` (e.g. `/movie/M2024_01`)
- **Response:**
  ```json
  {
    "movie_id": "M2024_01",
    "genre": "Sci-Fi",
    "release_year": 2024,
    "box_office_formatted": "$29.8M",
    "avg_rating": 8.5,
    "total_reviews": 4,
    "dominant_sentiment": "Positive",
    "pos_pct": 75.0,
    "neu_pct": 25.0,
    "neg_pct": 0.0,
    "reviews": [...]
  }
  ```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
