# 🎬 CineMetrics • Complete Source Code & Deployment Guide

This document contains the complete source code, architecture, and step-by-step startup instructions for the **CineMetrics Movie Sentiment, Genre & Movie ID Analytics Hub**.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Folder Structure](#-folder-structure)
3. [How to Start the Website](#-how-to-start-the-website)
4. [Complete Source Code](#-complete-source-code)
   - [1. requirements.txt](#1-requirementstxt)
   - [2. app.py](#2-apppy)
   - [3. templates/index.html](#3-templatesindexhtml)
   - [4. static/style.css](#4-staticstylecss)
   - [5. dataset.csv (Structure & Sample)](#5-datasetcsv-sample)
5. [Key Feature Breakdown](#-key-feature-breakdown)
   - [Movie ID Search & Intelligence Deep-Dive](#1-movie-id-search--intelligence-deep-dive)
   - [Dual Sentiment & Genre Engine](#2-dual-sentiment--genre-engine)
   - [Sarcasm & Irony Inverter](#3-sarcasm--irony-inverter)
   - [10-Year Interactive Graphs & Moving Data](#4-10-year-interactive-graphs--moving-data)
6. [API Reference](#-api-reference)
   - [POST /predict (Review Text Prompt)](#1-post-predict-review-text-prompt)
   - [POST /predict (Movie ID Lookup)](#2-post-predict-movie-id-lookup)
   - [GET /movie/<movie_id>](#3-get-moviemovie_id)
7. [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 📌 Project Overview

**CineMetrics** is a full-featured, real-time movie review analytics platform built with **Python & Flask**. It provides:
- **Movie ID Intelligence Lookup:** Type any Movie ID (e.g. `M2024_01`, `M2023_02`) to instantly see the film's **genre**, **sentiment consensus**, **number of reviews available**, **average rating**, **box office earnings**, and **complete list of individual reviews**.
- **Dual Sentiment & Genre Engine:** Analyzes review sentiment (Positive, Neutral, Negative) and simultaneously classifies film genre (Action, Sci-Fi, Horror, Thriller, Comedy, Romance, Drama, Animation) based on natural language semantic markers.
- **Sarcasm & Irony Detection:** Catches backhanded praise (*"Oh great, another two hours of my life I'll never get back"*) and inverts false-positive scores to negative.
- **10-Year Historical Trends (2015–2024):** Visualizes ratings evolution, box office revenue, and sentiment composition using interactive Chart.js graphs.
- **Moving Data Elements:** Smooth moving ticker tape ribbon (at comfortable 4x reduced pace), live streaming audience reviews simulation, and interactive 10-year timeline player.
- **Clickable Archive & Feature Search:** Click any Movie ID in the decade table or use the search bar to jump straight to in-depth analysis.

---

## 📁 Folder Structure

Ensure your files are arranged in the following directory layout:

```text
sentiment_analysis_app/
│
├── app.py                 # Core Flask backend, sentiment, genre & movie ID analyzer
├── dataset.csv            # 10-Year historical film reviews dataset (200 reviews, 50 movies)
├── requirements.txt       # Python dependencies
├── COMPLETE_DOCUMENTATION.md # Full documentation & reference
│
├── static/
│   └── style.css          # Dark-theme styling, 4x slow ticker ribbon, responsive UI
│
└── templates/
    └── index.html         # User interface with dual-view analytics & interactive charts
```

---

## 🚀 How to Start the Website

### Step 1: Open PowerShell or Command Prompt
Navigate to the project directory:
```powershell
cd C:\Users\HP\.gemini\antigravity\scratch\sentiment_analysis_app
```

### Step 2: Install Dependencies
Install Flask and NLTK using `pip`:
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Application
Launch the Flask development server:
```powershell
python app.py
```

You will see terminal output confirming:
```text
Starting Movie Analytics Hub on http://127.0.0.1:5000 ...
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Step 4: Open in Your Browser
Open any web browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)** (or `http://localhost:5000`)

---

## 💻 Complete Source Code

### 1. `requirements.txt`

```txt
Flask>=3.0.0
nltk>=3.8.0
```

---

### 2. `app.py`

```python
import os
import csv
import json
import re
from collections import defaultdict
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- SARCASM & IRONY DETECTION PATTERNS ---
SARCASM_PATTERNS = [
    # Ironic starters: "Oh great", "Oh wonderful", "Yeah, because..."
    r"\boh\s+(great|wonderful|fantastic|brilliant|joy|boy|thrill|lovely)\b",
    r"\byeah[, ]+(because|right|sure|as if|obviously)\b",
    r"\bjust what (i|we|the world) needed\b",
    r"\bwhat a (pleasant )?surprise\b",
    r"\bthanks for wasting\b",
    
    # Conditional irony: "[Praise word] ... if you (have no brain|consider X an actor|enjoy being bored)"
    r"\b(masterpiece|brilliant|great|amazing|groundbreaking|genius|stellar|award-winning)\b.*?\bif\s+(you|one)\b.*?\b(no brain|consider|asleep|torture|low standard|blind|deaf|zero standards?)\b",
    r"\b(if|unless)\s+(you|one)\s+consider\s+\w+\s+(an?\s+)?(actor|actress|plot|movie|film|cinema|masterpiece)\b",
    r"\b(if|unless)\s+(you|one)\s+(have|has)\s+(no|zero)\s+(taste|brain|standards?|expectations?)\b",

    # Contrastive dissonance: positive verb + negative experience
    r"\b(loved?|enjoyed?|liked?)\b.*?\b(sitting in the dark|being bored|bored to death|wasting|suffering|sleeping|headache|painful)\b",
    r"\bmasterpiece of (terrible|awful|garbage|bad|poor|trash|mediocrity|boredom|cringe)\b",
    r"\bwork of art in (terrible|awful|crap|garbage|bad|boring)\b",
    
    # Sarcastic idioms
    r"\b(two|three|\d+)\s+hours?\s+of\s+my\s+life\s+(i|i'll|i will)\s+never\s+get\s+back\b",
    r"\bwatching paint dry\b",
    r"\bcure for insomnia\b",
    r"\bput\s+(me|us|everyone)\s+to\s+sleep\b",
    r"\b10/10\s+for\s+(putting me to sleep|boredom|wasting time|terrible acting)\b"
]

def detect_sarcasm(text):
    clean = text.lower().strip()
    for pattern in SARCASM_PATTERNS:
        match = re.search(pattern, clean)
        if match:
            return True, match.group(0)
            
    # Scare quotes around praise words with negative context: What a "masterpiece"
    if re.search(r'[\"\'“](masterpiece|brilliant|genius|amazing|classic)[\"\'”]', clean):
        if any(neg in clean for neg in ['not', 'terrible', 'bad', 'awful', 'joke', 'disappointing', 'pretentious']):
            return True, 'scare quotes irony'
            
    return False, None
            
# --- GENRE CLASSIFICATION ENGINE ---
GENRE_RULES = {
    'Sci-Fi': {
        'emoji': '🚀',
        'keywords': ['sci-fi', 'scifi', 'sci fi', 'science fiction', 'space', 'spaceship', 'spaceships', 'alien', 'aliens', 'futuristic', 'time travel', 'galaxy', 'robot', 'robots', 'ai', 'cyberpunk', 'interstellar', 'universe', 'dystopian', 'quantum', 'parallel universe', 'starlight', 'mars', 'orbit', 'solar', 'solaris', 'technology', 'mind-bending', 'mind bending', 'mind-blending', 'mind blending']
    },
    'Action': {
        'emoji': '💥',
        'keywords': ['action', 'fight', 'fights', 'fighting', 'explosion', 'explosions', 'car chase', 'car chases', 'chase', 'stunt', 'stunts', 'martial arts', 'superhero', 'superheroes', 'combat', 'blast', 'adrenaline', 'shootout', 'shootouts', 'gun', 'guns', 'weapon', 'weapons', 'velocity', 'heist', 'battle', 'warrior', 'warriors', 'overdrive']
    },
    'Horror': {
        'emoji': '👻',
        'keywords': ['horror', 'scary', 'terrifying', 'jump scare', 'jump scares', 'haunted', 'ghost', 'ghosts', 'demon', 'demons', 'blood', 'bloody', 'creepy', 'monster', 'monsters', 'nightmare', 'nightmares', 'possessed', 'slasher', 'eerie', 'chilling', 'fear', 'sinister', 'woods', 'gore', 'corridor', 'darkness', 'spooky']
    },
    'Thriller': {
        'emoji': '🔍',
        'keywords': ['thriller', 'thrillers', 'suspense', 'mystery', 'mysteries', 'plot twist', 'plot twists', 'twist', 'twists', 'killer', 'killers', 'murder', 'murders', 'detective', 'detectives', 'investigation', 'investigations', 'conspiracy', 'psychological', 'edge of your seat', 'suspect', 'suspects', 'whodunit', 'interrogation', 'shadow', 'midnight']
    },
    'Comedy': {
        'emoji': '😂',
        'keywords': ['comedy', 'comedies', 'hilarious', 'funny', 'laugh', 'laughs', 'laughing', 'humor', 'joke', 'jokes', 'chuckle', 'parody', 'sitcom', 'satire', 'goofy', 'witty', 'comic timing', 'amusing', 'silly', 'neighbors']
    },
    'Romance': {
        'emoji': '❤️',
        'keywords': ['romance', 'romantic', 'love story', 'love', 'passion', 'lover', 'lovers', 'dating', 'couple', 'couples', 'affection', 'sweetheart', 'chemistry', 'heartwarming', 'relationship', 'relationships', 'venice', 'letters to venice', 'summer forever', 'wedding']
    },
    'Drama': {
        'emoji': '🎭',
        'keywords': ['drama', 'dramas', 'emotional', 'heartbreak', 'tearjerker', 'touching', 'poignant', 'grief', 'struggle', 'realistic', 'inspiring', 'character development', 'human condition', 'tragedy', 'acting', 'performances', 'dialogue', 'storyline', 'classic', 'horizon']
    },
    'Animation': {
        'emoji': '🎨',
        'keywords': ['animation', 'animated', 'cartoon', 'cartoons', 'cgi', 'pixar', 'disney', 'anime', 'voice acting', 'illustration', 'illustrations', 'kingdom of clouds', 'family-friendly', 'kids']
    }
}

def detect_genre(text):
    clean = text.lower().strip()
    scores = {}
    matches_found = {}
    
    for genre, data in GENRE_RULES.items():
        score = 0
        found = []
        for kw in data['keywords']:
            pattern = r'\b' + re.escape(kw) + r'\b'
            matches = re.findall(pattern, clean)
            if matches:
                # Direct genre name mentioned gives 3x weight
                weight = 3 if kw.rstrip('s') in genre.lower() else 1
                score += len(matches) * weight
                found.append(kw)
        if score > 0:
            scores[genre] = score
            matches_found[genre] = list(set(found))
            
    if not scores:
        return {
            'genre': 'General / Cinematic',
            'emoji': '🎬',
            'confidence': 'Moderate',
            'clues': ['General cinematic tone']
        }
        
    sorted_genres = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_genre, top_score = sorted_genres[0]
    total_score = sum(scores.values())
    confidence_pct = min(98, max(60, int((top_score / total_score) * 100)))
    
    return {
        'genre': top_genre,
        'emoji': GENRE_RULES[top_genre]['emoji'],
        'confidence': f'{confidence_pct}%',
        'clues': matches_found[top_genre][:4]
    }


# --- SENTIMENT ANALYZER (NLTK VADER + SARCASM & CINEMA TUNING LAYER) ---
CINEMA_LEXICON = {
    # Sci-Fi praise
    'mind-bending': 3.0,
    'mindbending': 3.0,
    'mind-blending': 3.0,
    'mindblending': 3.0,
    'blending': 0.8,
    'bending': 0.8,
    'sci-fi': 1.8,
    'scifi': 1.8,
    'spaceships': 1.5,
    'spaceship': 1.5,
    'futuristic': 1.5,
    'interstellar': 2.0,
    'breathtaking': 3.2,
    'spectacular': 3.0,

    # Romance & Drama praise
    'romantic': 2.8,
    'romance': 2.8,
    'heartwarming': 3.2,
    'touching': 2.8,
    'poignant': 2.6,
    'heartfelt': 2.8,
    'tender': 2.2,
    'chemistry': 2.0,
    'emotional': 2.0,
    'tearjerker': 1.8,
    'unforgettable': 2.8,
    'gripping': 2.6,
    'captivating': 2.8,
    'thrilling': 2.6,
    'compelling': 2.5,
    'stunning': 3.0,
    'charming': 2.6,

    # Horror cinema praise (which general NLP mistakenly scores as negative)
    'horror': 2.0,
    'scary': 2.4,
    'terrifying': 2.8,
    'chilling': 2.2,
    'creepy': 2.2,
    'spooky': 2.0,
    'eerie': 2.0,
    'haunted': 1.8,
    'sinister': 1.8,
    'demon': 1.2,
    'demons': 1.2,
    'monster': 1.5,
    'monsters': 1.5,

    # Thriller cinema praise
    'suspense': 2.5,
    'suspenseful': 2.8,
    'intense': 2.2,
    'twist': 2.2,
    'twists': 2.2,
    'mystery': 2.0,
    'mysteries': 2.0,
    'shocking': 1.8,
    'thriller': 2.2,
    'thrillers': 2.2,
}

CRITICAL_NEGATIVES = [
    r'\bwaste\b', r'\bterrible\b', r'\bawful\b', r'\bhorrible\b', r'\bboring\b',
    r'\bbored\b', r'\bdisappoint(ing|ed)?\b', r'\bruined\b', r'\bpoor\b',
    r'\bgarbage\b', r'\btrash\b', r'\bcringe\b', r'\bworst\b', r'\bunwatchable\b',
    r'\blacks?\s+soul\b', r'\bplot\s+holes?\b', r'\bcheap\b', r'\bfail(ed)?\b',
    r'\bnot\s+scary\b', r'\bnever\s+scary\b', r'\blaughable\b', r'\blame\b',
    r'\bdull\b', r'\buninspired\b', r'\bunconvincing\b'
]

CINEMA_PRAISE_PATTERNS = [
    r'\bmind[- ]?(bending|blending)\b', r'\bsci[- ]?fi\b', r'\bspaceships?\b',
    r'\bromantic\b', r'\bromance\b', r'\btouching\b', r'\bheartwarming\b',
    r'\bemotional\b', r'\bpoignant\b', r'\bheartfelt\b', r'\bcharming\b',
    r'\bterrifying\b', r'\bscary\b', r'\bcreep(y|ing)\b', r'\beerie\b',
    r'\bspooky\b', r'\bhaunted\b', r'\bjump\s+scares?\b', r'\bsinister\b',
    r'\bchilling\b', r'\bmonsters?\b', r'\bdemons?\b', r'\bhorror\b',
    r'\bsuspense(ful)?\b', r'\bmystery\b', r'\bthrill(er|ing)?\b', r'\bshocking\b'
]

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
        
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(CINEMA_LEXICON)
    
    def analyze_sentiment(text):
        scores = sia.polarity_scores(text)
        compound = scores['compound']
        is_sarcastic, reason = detect_sarcasm(text)
        clean = text.lower()
        
        if is_sarcastic:
            # Invert polarity because positive terms were used sarcastically
            if compound >= 0:
                compound = -max(0.45, round(abs(compound), 4))
            sentiment = "Negative"
            badge_label = "Negative (Sarcastic)"
            emoji = "🙃"
            color = "danger"
            explanation = f"Sarcasm & irony detected in phrasing ('{reason}'). True sentiment inverted to Negative."
            pos_pct = 5.0
            neu_pct = round(scores['neu'] * 100, 1)
            neg_pct = 95.0
        else:
            # Cinema domain calibration: praise in horror, thriller, sci-fi, romance, drama
            has_critical_neg = any(re.search(pat, clean) for pat in CRITICAL_NEGATIVES)
            has_cinema_praise = any(re.search(pat, clean) for pat in CINEMA_PRAISE_PATTERNS)
            
            if has_cinema_praise and not has_critical_neg:
                if compound < 0.25:
                    compound = max(0.55, round(compound + 0.65, 4))
                    scores['pos'] = max(0.65, scores['pos'] + 0.5)
                    scores['neg'] = 0.0
                    scores['neu'] = round(max(0.0, 1.0 - scores['pos']), 3)

            if compound >= 0.05:
                sentiment = "Positive"
                badge_label = "Positive"
                emoji = "😊"
                color = "success"
            elif compound <= -0.05:
                sentiment = "Negative"
                badge_label = "Negative"
                emoji = "😞"
                color = "danger"
            else:
                sentiment = "Neutral"
                badge_label = "Neutral"
                emoji = "😐"
                color = "warning"
            explanation = None
            pos_pct = round(scores['pos'] * 100, 1)
            neu_pct = round(scores['neu'] * 100, 1)
            neg_pct = round(scores['neg'] * 100, 1)
            
        genre_info = detect_genre(text)
        return {
            "text": text,
            "sentiment": sentiment,
            "badge_label": badge_label,
            "emoji": emoji,
            "color": color,
            "compound": round(compound, 4),
            "is_sarcastic": is_sarcastic,
            "explanation": explanation,
            "genre_detected": genre_info['genre'],
            "genre_emoji": genre_info['emoji'],
            "genre_confidence": genre_info['confidence'],
            "genre_clues": genre_info['clues'],
            "pos": pos_pct,
            "neu": neu_pct,
            "neg": neg_pct
        }

except Exception:
    POS_WORDS = {
        "amazing", "good", "masterpiece", "great", "excellent", "brilliant", "visually",
        "spectacular", "romantic", "romance", "mind-bending", "mind-blending", "touching",
        "heartwarming", "emotional", "poignant", "charming", "sci-fi", "scifi", "scary",
        "terrifying", "creepy", "spooky", "eerie", "haunted", "chilling", "sinister",
        "horror", "suspense", "thriller", "mystery", "thrilling"
    }
    NEG_WORDS = {
        "waste", "bad", "terrible", "boring", "awful", "horrible", "ruined", "disappointing",
        "laughable", "lame", "dull", "uninspired", "unconvincing"
    }

    def analyze_sentiment(text):
        is_sarcastic, reason = detect_sarcasm(text)
        clean = text.lower()
        words = set(clean.replace("!", "").replace(".", "").replace(",", "").split())
        pos_hits = len(words.intersection(POS_WORDS))
        neg_hits = len(words.intersection(NEG_WORDS))
        genre_info = detect_genre(text)
        
        has_critical_neg = any(re.search(pat, clean) for pat in CRITICAL_NEGATIVES)
        has_cinema_praise = any(re.search(pat, clean) for pat in CINEMA_PRAISE_PATTERNS)
        
        if is_sarcastic:
            sentiment, badge_label, emoji, color, compound = "Negative", "Negative (Sarcastic)", "🙃", "danger", -0.55
            explanation = f"Sarcasm detected ('{reason}'). Adjusted to Negative."
        elif has_cinema_praise and not has_critical_neg:
            sentiment, badge_label, emoji, color, compound, explanation = "Positive", "Positive", "😊", "success", 0.65, None
        elif pos_hits > neg_hits:
            sentiment, badge_label, emoji, color, compound, explanation = "Positive", "Positive", "😊", "success", 0.65, None
        elif neg_hits > pos_hits:
            sentiment, badge_label, emoji, color, compound, explanation = "Negative", "Negative", "😞", "danger", -0.55, None
        else:
            sentiment, badge_label, emoji, color, compound, explanation = "Neutral", "Neutral", "😐", "warning", 0.0, None
            
        return {
            "text": text, "sentiment": sentiment, "badge_label": badge_label, "emoji": emoji,
            "color": color, "compound": compound, "is_sarcastic": is_sarcastic, "explanation": explanation,
            "genre_detected": genre_info['genre'], "genre_emoji": genre_info['emoji'],
            "genre_confidence": genre_info['confidence'], "genre_clues": genre_info['clues'],
            "pos": 75.0 if sentiment == "Positive" else 5.0,
            "neu": 100.0 if sentiment == "Neutral" else 25.0,
            "neg": 75.0 if sentiment == "Negative" else 0.0
        }

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset.csv')

def load_and_analyze_dataset():
    reviews = []
    if not os.path.exists(DATASET_PATH):
        return reviews
        
    with open(DATASET_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            analysis = analyze_sentiment(row['review_text'])
            row['sentiment'] = analysis['sentiment']
            row['badge_label'] = analysis['badge_label']
            row['compound'] = analysis['compound']
            row['emoji'] = analysis['emoji']
            row['color'] = analysis['color']
            row['is_sarcastic'] = analysis['is_sarcastic']
            row['rating'] = float(row['rating'])
            row['release_year'] = int(row['release_year'])
            row['box_office_earnings'] = int(row['box_office_earnings'])
            reviews.append(row)
    return reviews


def build_analytics(dataset):
    by_year = defaultdict(list)
    by_genre = defaultdict(list)
    
    for r in dataset:
        by_year[r['release_year']].append(r)
        by_genre[r['genre']].append(r)
        
    years = sorted(by_year.keys())
    yearly_labels = [str(y) for y in years]
    avg_ratings_by_year = []
    avg_box_office_by_year = []
    positive_by_year = []
    neutral_by_year = []
    negative_by_year = []
    
    for y in years:
        revs = by_year[y]
        count = len(revs)
        avg_r = round(sum(x['rating'] for x in revs) / count, 2)
        avg_bo = round(sum(x['box_office_earnings'] for x in revs) / (count * 1_000_000), 2)
        
        pos = sum(1 for x in revs if x['sentiment'] == 'Positive')
        neu = sum(1 for x in revs if x['sentiment'] == 'Neutral')
        neg = sum(1 for x in revs if x['sentiment'] == 'Negative')
        
        avg_ratings_by_year.append(avg_r)
        avg_box_office_by_year.append(avg_bo)
        positive_by_year.append(pos)
        neutral_by_year.append(neu)
        negative_by_year.append(neg)
        
    genres = sorted(by_genre.keys())
    genre_ratings = []
    genre_counts = []
    genre_pos_pct = []
    
    for g in genres:
        revs = by_genre[g]
        g_count = len(revs)
        g_avg_r = round(sum(x['rating'] for x in revs) / g_count, 1)
        g_pos = sum(1 for x in revs if x['sentiment'] == 'Positive')
        
        genre_ratings.append(g_avg_r)
        genre_counts.append(g_count)
        genre_pos_pct.append(round((g_pos / g_count) * 100, 1))
        
    total = len(dataset)
    stats = {
        'total': total,
        'positive': sum(1 for r in dataset if r['sentiment'] == 'Positive'),
        'neutral': sum(1 for r in dataset if r['sentiment'] == 'Neutral'),
        'negative': sum(1 for r in dataset if r['sentiment'] == 'Negative'),
        'avg_rating': round(sum(r['rating'] for r in dataset) / total, 1) if total > 0 else 0,
        'year_range': f"{years[0]} – {years[-1]}" if years else "Last 10 Years",
        'years_count': len(years)
    }
    
    charts_data = {
        "yearly": {
            "years": yearly_labels,
            "ratings": avg_ratings_by_year,
            "box_office": avg_box_office_by_year,
            "positive": positive_by_year,
            "neutral": neutral_by_year,
            "negative": negative_by_year
        },
        "genre": {
            "genres": genres,
            "ratings": genre_ratings,
            "counts": genre_counts,
            "pos_pct": genre_pos_pct
        }
    }
    
    ticker_items = dataset[::3] if len(dataset) > 20 else dataset
    return stats, charts_data, ticker_items


def get_movie_analysis(movie_id):
    """Analyze all reviews for a specific movie ID, computing genre, sentiment, and review statistics."""
    dataset = load_and_analyze_dataset()
    m_id_clean = movie_id.strip().upper()
    matching_reviews = [r for r in dataset if r['movie_id'].upper() == m_id_clean]
    
    if not matching_reviews:
        return None
        
    genre = matching_reviews[0]['genre']
    year = matching_reviews[0]['release_year']
    bo_val = float(matching_reviews[0]['box_office_earnings'])
    bo_formatted = f"${bo_val / 1_000_000:.1f}M"
    
    total_reviews = len(matching_reviews)
    avg_rating = round(sum(r['rating'] for r in matching_reviews) / total_reviews, 1)
    
    pos_count = sum(1 for r in matching_reviews if r['sentiment'] == 'Positive')
    neu_count = sum(1 for r in matching_reviews if r['sentiment'] == 'Neutral')
    neg_count = sum(1 for r in matching_reviews if r['sentiment'] == 'Negative')
    
    pos_pct = round((pos_count / total_reviews) * 100, 1)
    neu_pct = round((neu_count / total_reviews) * 100, 1)
    neg_pct = round((neg_count / total_reviews) * 100, 1)
    
    if pos_count >= max(neu_count, neg_count):
        dominant_sentiment = "Positive"
        dominant_emoji = "😊"
        dominant_color = "success"
    elif neg_count >= max(pos_count, neu_count):
        dominant_sentiment = "Negative"
        dominant_emoji = "😞"
        dominant_color = "danger"
    else:
        dominant_sentiment = "Neutral"
        dominant_emoji = "😐"
        dominant_color = "warning"
        
    avg_compound = round(sum(r['compound'] for r in matching_reviews) / total_reviews, 4)
    
    reviews_list = []
    for r in matching_reviews:
        reviews_list.append({
            'review_id': r['review_id'],
            'user_id': r['user_id'],
            'rating': r['rating'],
            'review_text': r['review_text'],
            'sentiment': r['sentiment'],
            'badge_label': r['badge_label'],
            'emoji': r['emoji'],
            'color': r['color'],
            'compound': r['compound'],
            'is_sarcastic': r.get('is_sarcastic', False)
        })
        
    return {
        "movie_id": m_id_clean,
        "genre": genre,
        "genre_emoji": GENRE_RULES.get(genre, {}).get('emoji', '🎬'),
        "release_year": year,
        "box_office_earnings": bo_val,
        "box_office_formatted": bo_formatted,
        "total_reviews": total_reviews,
        "avg_rating": avg_rating,
        "pos_count": pos_count,
        "neu_count": neu_count,
        "neg_count": neg_count,
        "pos_pct": pos_pct,
        "neu_pct": neu_pct,
        "neg_pct": neg_pct,
        "dominant_sentiment": dominant_sentiment,
        "dominant_emoji": dominant_emoji,
        "dominant_color": dominant_color,
        "avg_compound": avg_compound,
        "reviews": reviews_list
    }


@app.route('/')
def home():
    dataset = load_and_analyze_dataset()
    stats, charts_data, ticker_items = build_analytics(dataset)
    unique_movie_ids = sorted(list(set(r['movie_id'] for r in dataset)))
    return render_template(
        'index.html',
        stats=stats,
        dataset=dataset,
        charts_data_json=json.dumps(charts_data),
        ticker_items=ticker_items,
        unique_movie_ids=unique_movie_ids
    )


@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        review_text = data.get('review_text', '').strip()
    else:
        review_text = request.form.get('review_text', '').strip()
        
    if not review_text:
        return jsonify({"error": "Please provide a review text or Movie ID."}), 400
        
    # Check if input matches or contains a Movie ID format (e.g. M2024_01, M2020_02, M001)
    m_match = re.search(r'\b(M\d{4}_\d{2}|M\d+)\b', review_text, re.IGNORECASE)
    if m_match and len(review_text) <= 30:
        extracted_id = m_match.group(1).upper()
        movie_data = get_movie_analysis(extracted_id)
        if movie_data:
            return jsonify({
                "is_movie_id": True,
                "movie_data": movie_data
            })
            
    result = analyze_sentiment(review_text)
    result["is_movie_id"] = False
    
    # Check if review text matches an existing movie in dataset for seamless alignment
    dataset = load_and_analyze_dataset()
    clean_input = review_text.lower().strip()
    if len(clean_input) >= 10:
        for r in dataset:
            if clean_input in r['review_text'].lower() or r['review_text'].lower() in clean_input:
                result["matched_movie_id"] = r['movie_id']
                result["matched_movie_rating"] = r['rating']
                result["matched_movie_year"] = r['release_year']
                result["matched_movie_genre"] = r['genre']
                break
                
    return jsonify(result)


@app.route('/movie/<movie_id>')
def movie_details(movie_id):
    """Dedicated endpoint to fetch complete analytics for any movie ID."""
    movie_data = get_movie_analysis(movie_id)
    if not movie_data:
        return jsonify({"error": f"Movie ID '{movie_id}' not found in database."}), 404
    return jsonify(movie_data)


if __name__ == '__main__':
    print("Starting Movie Analytics Hub on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=True)

```

---

### 3. `templates/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CineMetrics • Movie Sentiment & 10-Year Rating Intelligence</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <!-- Chart.js CDN for Interactive Graphs -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>

    <!-- Moving Data Ticker Ribbon (Half Speed) -->
    <div class="ticker-wrap">
        <div class="ticker-content" id="tickerTrack">
            {% for item in ticker_items %}
            <div class="ticker-item">
                <span class="ticker-tag">{{ item.movie_id }}</span>
                <span class="ticker-rating">⭐ {{ item.rating }}/10</span>
                <span class="ticker-text">"{{ item.review_text }}"</span>
                <span class="ticker-badge badge-sentiment {{ item.color }}">{{ item.emoji }} {{ item.sentiment }}</span>
                <span style="color: #94a3b8; font-size: 0.75rem;">${{ "{:,.1f}".format(item.box_office_earnings / 1000000) }}M</span>
            </div>
            {% endfor %}
            <!-- Duplicated items for seamless continuous loop -->
            {% for item in ticker_items %}
            <div class="ticker-item">
                <span class="ticker-tag">{{ item.movie_id }}</span>
                <span class="ticker-rating">⭐ {{ item.rating }}/10</span>
                <span class="ticker-text">"{{ item.review_text }}"</span>
                <span class="ticker-badge badge-sentiment {{ item.color }}">{{ item.emoji }} {{ item.sentiment }}</span>
                <span style="color: #94a3b8; font-size: 0.75rem;">${{ "{:,.1f}".format(item.box_office_earnings / 1000000) }}M</span>
            </div>
            {% endfor %}
        </div>
    </div>

    <div class="container">
        <!-- Header -->
        <header>
            <div class="pulse-indicator">
                <span class="dot-live"></span>
                <span>REAL-TIME AUDIENCE INTELLIGENCE</span>
            </div>
            <h1>🎬 CineMetrics • Movie Sentiment & Rating Analytics Hub</h1>
            <p>Real-Time Audience Sentiment & 10-Year Film Performance Intelligence ({{ stats.year_range }})</p>
        </header>

        <!-- Stats Overview Cards -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value color-gold">{{ stats.years_count }} Years</div>
                <div class="stat-label">Historical Span</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ stats.total }}</div>
                <div class="stat-label">Total Reviews Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value color-pos">{{ stats.positive }}</div>
                <div class="stat-label">Positive Sentiment 😊</div>
            </div>
            <div class="stat-card">
                <div class="stat-value color-neu">{{ stats.neutral }}</div>
                <div class="stat-label">Neutral Sentiment 😐</div>
            </div>
            <div class="stat-card">
                <div class="stat-value color-neg">{{ stats.negative }}</div>
                <div class="stat-label">Negative Sentiment 😞</div>
            </div>
            <div class="stat-card">
                <div class="stat-value color-rate">{{ stats.avg_rating }} ⭐</div>
                <div class="stat-label">Decade Avg Rating</div>
            </div>
        </div>

        <!-- 1. FEATURE SEARCH: SENTIMENT & GENRE INTELLIGENCE (DISPLAYED FIRST) -->
        <div class="card" id="featureSearchSection">
            <div class="card-header-flex">
                <h2>🔍 Feature Search: Sentiment & Genre Classification</h2>
                <span class="pulse-indicator" style="font-size: 0.75rem; padding: 0.2rem 0.6rem;">
                    <span class="dot-live"></span> DUAL SENTIMENT &amp; GENRE ENGINE
                </span>
            </div>
            <p style="color: var(--text-secondary); font-size: 0.88rem; margin-bottom: 1.1rem;">
                Enter any movie review or story prompt. The engine simultaneously evaluates <strong>audience sentiment</strong> and classifies <strong>which film genre</strong> (Action, Sci-Fi, Horror, Thriller, Comedy, Romance, Drama, Animation) the movie belongs to based on semantic context clues.
            </p>
            
            <div class="analyzer-grid">
                <div>
                    <form id="sentimentForm">
                        <div class="input-group">
                            <label style="font-size: 0.88rem; color: var(--text-secondary); font-weight: 600;">
                                Enter Movie Review / Story Prompt:
                            </label>
                            <textarea id="reviewInput" placeholder="Type or paste any audience feedback, plot premise, or film review to determine both its sentiment and genre..." required></textarea>
                            
                            <!-- Standard Sentiment Samples -->
                            <div class="sample-prompts">
                                <span>Sentiment Presets:</span>
                                <button type="button" class="sample-btn" onclick="setSample('Masterpiece! A timeless classic.')">Masterpiece!</button>
                                <button type="button" class="sample-btn" onclick="setSample('Amazing movie! Kept me hooked until the final scene.')">Amazing movie!</button>
                                <button type="button" class="sample-btn" onclick="setSample('Good plot, strong performances, and solid pacing.')">Good plot.</button>
                                <button type="button" class="sample-btn" onclick="setSample('Average film. Enjoyable enough for a casual watch.')">Average.</button>
                                <button type="button" class="sample-btn" onclick="setSample('Waste of time. Disappointing sequel that lacks soul.')">Waste of time.</button>
                            </div>

                            <!-- Genre Clarity Test Prompts -->
                            <div class="sample-prompts" style="margin-top: 0.25rem;">
                                <span style="color: #38bdf8; font-weight: 600;">Genre Clarity Prompts 🏷️:</span>
                                <button type="button" class="sample-btn" style="border-color: rgba(56, 189, 248, 0.4); color: #38bdf8;" onclick="setSample('Mind-bending sci-fi with spaceships, time travel, and alien worlds.')">🚀 Sci-Fi</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(56, 189, 248, 0.4); color: #38bdf8;" onclick="setSample('High-octane car chases, explosive combat, and thrilling martial arts stunts!')">💥 Action</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(56, 189, 248, 0.4); color: #38bdf8;" onclick="setSample('Terrifying haunted house with eerie jump scares and sinister demons.')">👻 Horror</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(56, 189, 248, 0.4); color: #38bdf8;" onclick="setSample('Hilarious jokes, laugh-out-loud humor, and witty comic timing.')">😂 Comedy</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(56, 189, 248, 0.4); color: #38bdf8;" onclick="setSample('Intense murder mystery with psychological suspense and a shocking twist.')">🔍 Thriller</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(56, 189, 248, 0.4); color: #38bdf8;" onclick="setSample('Heartwarming romance with tender love and undeniable chemistry.')">❤️ Romance</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(56, 189, 248, 0.4); color: #38bdf8;" onclick="setSample('A touching emotional drama exploring heartbreak and family struggles.')">🎭 Drama</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(56, 189, 248, 0.4); color: #38bdf8;" onclick="setSample('Whimsical animated adventure with colorful illustrations and charming voice acting.')">🎨 Animation</button>
                            </div>

                            <!-- Sarcasm Test Prompts -->
                            <div class="sample-prompts" style="margin-top: 0.25rem;">
                                <span style="color: #f472b6; font-weight: 600;">Sarcasm Tests 🙃:</span>
                                <button type="button" class="sample-btn" style="border-color: rgba(244, 114, 182, 0.4); color: #f472b6;" onclick="setSample('Oh great, another two hours of my life I will never get back.')">"Oh great, another two hours..."</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(244, 114, 182, 0.4); color: #f472b6;" onclick="setSample('Brilliant movie, if you have no brain cells left.')">"Brilliant, if you have no brain..."</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(244, 114, 182, 0.4); color: #f472b6;" onclick="setSample('I just loved sitting in the dark for 3 hours being utterly bored.')">"Loved being bored for 3 hours..."</button>
                            </div>

                            <!-- Movie ID Search / Lookup Presets -->
                            <div class="sample-prompts" style="margin-top: 0.25rem;">
                                <span style="color: #fbbf24; font-weight: 600;">Movie ID Lookups 🎬:</span>
                                <button type="button" class="sample-btn" style="border-color: rgba(251, 191, 36, 0.4); color: #fbbf24;" onclick="lookupMovieId('M2024_01')">🎬 M2024_01 (Action)</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(251, 191, 36, 0.4); color: #fbbf24;" onclick="lookupMovieId('M2024_02')">🎬 M2024_02 (Drama)</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(251, 191, 36, 0.4); color: #fbbf24;" onclick="lookupMovieId('M2023_01')">🎬 M2023_01 (Romance)</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(251, 191, 36, 0.4); color: #fbbf24;" onclick="lookupMovieId('M2018_02')">🎬 M2018_02 (Sci-Fi)</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(251, 191, 36, 0.4); color: #fbbf24;" onclick="lookupMovieId('M2024_03')">🎬 M2024_03 (Thriller)</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(251, 191, 36, 0.4); color: #fbbf24;" onclick="lookupMovieId('M2016_04')">🎬 M2016_04 (Comedy)</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(251, 191, 36, 0.4); color: #fbbf24;" onclick="lookupMovieId('M2017_02')">🎬 M2017_02 (Horror)</button>
                                <button type="button" class="sample-btn" style="border-color: rgba(251, 191, 36, 0.4); color: #fbbf24;" onclick="lookupMovieId('M2017_04')">🎬 M2017_04 (Animation)</button>
                            </div>

                            <button type="submit" class="btn-primary" id="analyzeBtn" style="margin-top: 0.6rem;">Analyze Sentiment &amp; Detect Genre / Movie ID</button>
                        </div>
                    </form>

                    <!-- Feature Keyword Search across Movie Archive -->
                    <div class="search-box-wrap">
                        <label style="font-size: 0.88rem; color: var(--text-secondary); font-weight: 600; display: block; margin-bottom: 0.4rem;">
                            Search 10-Year Archive by Movie ID, Genre, Keyword, Title, or Year:
                        </label>
                        
                        <!-- Genre Filter Buttons for Search -->
                        <div class="sample-prompts" style="margin-bottom: 0.5rem;">
                            <span style="font-size: 0.78rem;">Quick Genre Filter:</span>
                            <button type="button" class="sample-btn" onclick="quickFilterGenre('')">All</button>
                            <button type="button" class="sample-btn" onclick="quickFilterGenre('Action')">💥 Action</button>
                            <button type="button" class="sample-btn" onclick="quickFilterGenre('Sci-Fi')">🚀 Sci-Fi</button>
                            <button type="button" class="sample-btn" onclick="quickFilterGenre('Drama')">🎭 Drama</button>
                            <button type="button" class="sample-btn" onclick="quickFilterGenre('Thriller')">🔍 Thriller</button>
                            <button type="button" class="sample-btn" onclick="quickFilterGenre('Comedy')">😂 Comedy</button>
                            <button type="button" class="sample-btn" onclick="quickFilterGenre('Romance')">❤️ Romance</button>
                            <button type="button" class="sample-btn" onclick="quickFilterGenre('Horror')">👻 Horror</button>
                            <button type="button" class="sample-btn" onclick="quickFilterGenre('Animation')">🎨 Animation</button>
                        </div>

                        <span class="search-icon-pos">🔍</span>
                        <input 
                            type="text" 
                            id="featureSearchInput" 
                            class="search-input" 
                            placeholder="Type Movie ID (e.g. M2024_01) or keyword/genre to filter archive..." 
                            oninput="handleFeatureSearch(this.value)">
                        <div id="searchResultCount" style="font-size: 0.8rem; color: var(--accent-blue); margin-top: 0.4rem;"></div>
                    </div>
                </div>

                <div>
                    <div id="resultContainer" class="result-box" style="display: none; height: 100%;">
                        
                        <!-- VIEW 1: MOVIE ID DEEP-DIVE ANALYSIS VIEW -->
                        <div id="movieAnalysisView" style="display: none;">
                            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 0.65rem; margin-bottom: 0.85rem;">
                                <div>
                                    <span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">🎬 Movie ID Intelligence Report</span>
                                    <div style="font-size: 1.4rem; font-weight: 800; color: #fbbf24; display: flex; align-items: center; gap: 0.45rem; margin-top: 0.15rem;">
                                        <span id="mIdVal">M2024_01</span>
                                        <span id="mGenreBadge" style="font-size: 0.8rem; background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; padding: 0.15rem 0.5rem; border-radius: 999px;">Action</span>
                                    </div>
                                </div>
                                <div style="text-align: right;">
                                    <span style="font-size: 0.72rem; color: #94a3b8;">Release Year</span>
                                    <div id="mYearVal" style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">2024</div>
                                </div>
                            </div>

                            <!-- Key Metrics Grid for Movie ID -->
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-bottom: 0.85rem;">
                                <div style="background: rgba(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 8px; padding: 0.5rem; text-align: center;">
                                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;">Available Reviews</div>
                                    <div id="mReviewCount" style="font-size: 1.15rem; font-weight: 800; color: #38bdf8;">4 Reviews</div>
                                </div>
                                <div style="background: rgba(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 8px; padding: 0.5rem; text-align: center;">
                                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;">Avg Rating</div>
                                    <div id="mAvgRating" style="font-size: 1.15rem; font-weight: 800; color: #fbbf24;">8.5 ⭐</div>
                                </div>
                                <div style="background: rgba(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 8px; padding: 0.5rem; text-align: center;">
                                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;">Box Office</div>
                                    <div id="mBoxOffice" style="font-size: 1.15rem; font-weight: 800; color: #10b981;">$35.2M</div>
                                </div>
                            </div>

                            <!-- Sentiment Consensus Card -->
                            <div style="background: rgba(0,0,0,0.25); border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.85rem; border: 1px solid var(--border-color);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                                    <span style="font-size: 0.82rem; color: #cbd5e1;">Sentiment Consensus:</span>
                                    <span id="mDominantSentiment" class="badge-sentiment">Positive 😊</span>
                                </div>
                                <div style="display: flex; gap: 0.5rem; font-size: 0.76rem;">
                                    <span style="color: #10b981;" id="mPosReviews">3 Positive</span> • 
                                    <span style="color: #f59e0b;" id="mNeuReviews">1 Neutral</span> • 
                                    <span style="color: #ef4444;" id="mNegReviews">0 Negative</span>
                                </div>
                            </div>

                            <!-- List of All Available Reviews for this Movie -->
                            <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.4rem; text-transform: uppercase;">
                                All Reviews for this Movie (<span id="mListCount">4</span>):
                            </div>
                            <div id="mReviewsList" style="max-height: 180px; overflow-y: auto; display: flex; flex-direction: column; gap: 0.4rem; padding-right: 0.2rem;">
                                <!-- Populated dynamically by JS -->
                            </div>
                        </div>

                        <!-- VIEW 2: STANDARD REVIEW SENTIMENT & GENRE VIEW -->
                        <div id="reviewAnalysisView">
                            <!-- 1. Detected Film Genre Box -->
                            <div id="genreResultBox" style="background: rgba(56, 189, 248, 0.12); border: 1px solid #38bdf8; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.85rem;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div style="display: flex; align-items: center; gap: 0.45rem; font-size: 1.15rem; font-weight: 700;">
                                        <span id="resGenreEmoji">🎬</span>
                                        <span style="color: #f8fafc;">Detected Genre: <strong id="resGenre" style="color: #38bdf8;">General</strong></span>
                                    </div>
                                    <span id="resGenreConf" style="font-size: 0.78rem; background: #0f172a; padding: 0.2rem 0.6rem; border-radius: 999px; color: #38bdf8; font-weight: 600; border: 1px solid #38bdf8;">95% Confidence</span>
                                </div>
                                <div style="margin-top: 0.35rem; font-size: 0.82rem; color: #94a3b8;">
                                    Key Genre Clues: <span id="resGenreClues" style="color: #f1f5f9; font-weight: 500;">General film tone</span>
                                </div>
                            </div>

                            <!-- 2. Sentiment Classification Header -->
                            <div class="result-header">
                                <div class="sentiment-badge">
                                    <span id="resEmoji"></span>
                                    <span id="resSentiment"></span>
                                </div>
                                <div class="score-text">
                                    Polarity Score: <strong id="resCompound">0.0</strong>
                                </div>
                            </div>

                            <!-- 3. Sarcasm Warning Banner -->
                            <div id="sarcasmNotice" style="display: none; background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.85rem; font-size: 0.84rem; color: #fca5a5;">
                                <strong>🙃 Sarcasm / Irony Detected:</strong>
                                <div id="sarcasmReasonText" style="margin-top: 0.2rem; color: #fecaca;"></div>
                            </div>

                            <!-- 4. Component Breakdown -->
                            <div class="scores-breakdown">
                                <div class="score-item">
                                    Positive Component
                                    <span class="color-pos" id="resPos">0%</span>
                                </div>
                                <div class="score-item">
                                    Neutral Component
                                    <span class="color-neu" id="resNeu">0%</span>
                                </div>
                                <div class="score-item">
                                    Negative Component
                                    <span class="color-neg" id="resNeg">0%</span>
                                </div>
                            </div>

                            <!-- 5. Linked Archive Movie Banner -->
                            <div id="matchedArchiveMovieBanner" style="display: none; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 8px; padding: 0.65rem 0.85rem; margin-top: 0.85rem; font-size: 0.84rem;">
                                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                                    <div>
                                        <span style="color: #94a3b8;">Linked Archive Movie:</span>
                                        <strong id="matchedMovieTitle" style="color: #fbbf24; margin-left: 0.3rem;">M2024_01</strong>
                                        <span id="matchedMovieExtra" style="color: #94a3b8; font-size: 0.78rem; margin-left: 0.4rem;"></span>
                                    </div>
                                    <button type="button" id="matchedMovieBtn" class="sample-btn" style="font-size: 0.75rem; border-color: #38bdf8; color: #38bdf8; font-weight: 600;">View Movie Intelligence &rarr;</button>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div id="placeholderBox" style="border: 2px dashed var(--border-color); border-radius: 10px; height: 100%; min-height: 180px; display: flex; align-items: center; justify-content: center; color: var(--text-secondary); text-align: center; padding: 1.5rem;">
                        Enter a prompt, click a genre preset, or enter a Movie ID (e.g. M2024_01) to view complete sentiment and film metrics.
                    </div>
                </div>
            </div>
        </div>

        <!-- 2. 10-YEAR MOVIE DATABASE ARCHIVE (LOCATED DIRECTLY BELOW FEATURE & GENRE SEARCH) -->
        <div class="card" id="archiveTableSection">
            <div class="table-filter-bar">
                <h2>📋 Decade Film Archive ({{ dataset|length }} Reviews)</h2>
                <div class="filter-btn-group">
                    <button class="filter-btn active" onclick="filterTable('all', this)">All</button>
                    <button class="filter-btn" onclick="filterTable('Positive', this)">Positive</button>
                    <button class="filter-btn" onclick="filterTable('Neutral', this)">Neutral</button>
                    <button class="filter-btn" onclick="filterTable('Negative', this)">Negative</button>
                </div>
            </div>

            <div class="table-wrapper">
                <table id="reviewsTable">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Year</th>
                            <th>Movie ID</th>
                            <th>User ID</th>
                            <th>Rating</th>
                            <th>Review Text</th>
                            <th>Genre</th>
                            <th>Box Office</th>
                            <th>Sentiment</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in dataset %}
                        <tr data-sentiment="{{ row.sentiment }}" data-year="{{ row.release_year }}" data-movie-id="{{ row.movie_id }}" data-genre="{{ row.genre }}">
                            <td>{{ row.review_id }}</td>
                            <td><strong style="color: #fbbf24;">{{ row.release_year }}</strong></td>
                            <td><a href="javascript:void(0)" onclick="lookupMovieId('{{ row.movie_id }}')" title="Analyze movie {{ row.movie_id }}" style="color: #38bdf8; text-decoration: underline; font-weight: 600; cursor: pointer;">{{ row.movie_id }} 🔍</a></td>
                            <td>{{ row.user_id }}</td>
                            <td>⭐ {{ row.rating }}/10</td>
                            <td>{{ row.review_text }}</td>
                            <td>{{ row.genre }}</td>
                            <td>${{ "{:,.1f}".format(row.box_office_earnings / 1000000) }}M</td>
                            <td>
                                <span class="badge-sentiment {{ row.color }}">
                                    {{ row.emoji }} {{ row.sentiment }}
                                </span>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 3. 10-YEAR MOVIE RATING & BOX OFFICE GRAPHS -->
        <div class="charts-grid">
            <!-- Graph 1: 10-Year Average Rating Trend -->
            <div class="card">
                <div class="card-header-flex">
                    <h2>📈 10-Year Movie Rating & Box Office Trend</h2>
                    <div class="filter-btn-group">
                        <button class="filter-btn active" onclick="setChartFilter('all', this)">All 10 Years</button>
                        <button class="filter-btn" onclick="setChartFilter('recent', this)">Recent 5 Years</button>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="ratingsTrendChart"></canvas>
                </div>
            </div>

            <!-- Graph 2: 10-Year Annual Sentiment Composition -->
            <div class="card">
                <div class="card-header-flex">
                    <h2>🎭 Annual Sentiment Breakdown</h2>
                </div>
                <div class="chart-container">
                    <canvas id="sentimentBarChart"></canvas>
                </div>
            </div>
        </div>

        <!-- 3. 10-YEAR TIMELINE PLAYER (MOVING DATA FEATURE) -->
        <div class="timeline-card">
            <div class="timeline-header">
                <h3>🎞️ 10-Year Cinema Timeline Evolution</h3>
                <div class="timeline-controls">
                    <button class="play-btn" id="playTimelineBtn" onclick="toggleTimelinePlay()">
                        <span id="playIcon">▶</span> <span id="playText">Play 10-Year Evolution</span>
                    </button>
                    <button class="sample-btn" onclick="resetTimeline()">⏮ Reset</button>
                </div>
            </div>

            <!-- Interactive Year Chips -->
            <div class="year-chips" id="yearChips">
                <!-- Dynamically generated from JS -->
            </div>

            <!-- Dynamic Year Metrics Display -->
            <div class="timeline-display">
                <div>
                    <div class="timeline-metric-title">Active Release Year</div>
                    <div class="timeline-metric-val color-gold" id="dispYear">2015</div>
                </div>
                <div>
                    <div class="timeline-metric-title">Average Audience Rating</div>
                    <div class="timeline-metric-val color-rate" id="dispRating">7.8 / 10 ⭐</div>
                </div>
                <div>
                    <div class="timeline-metric-title">Average Box Office</div>
                    <div class="timeline-metric-val color-pos" id="dispBoxOffice">$18.5M</div>
                </div>
                <div>
                    <div class="timeline-metric-title">Sentiment Consensus</div>
                    <div class="timeline-metric-val" id="dispSentiment">😊 75% Positive</div>
                </div>
            </div>
        </div>

        <!-- 4. GENRE RATINGS & MOVING LIVE STREAM -->
        <div class="charts-grid">
            <!-- Graph 3: Genre Performance -->
            <div class="card">
                <div class="card-header-flex">
                    <h2>🏆 Decade Genre Ratings Performance</h2>
                </div>
                <div class="chart-container">
                    <canvas id="genreChart"></canvas>
                </div>
            </div>

            <!-- Moving Live Stream Feed -->
            <div class="card">
                <div class="card-header-flex">
                    <h2>📡 Live Audience Stream</h2>
                    <span class="pulse-indicator" style="font-size: 0.72rem; padding: 0.15rem 0.5rem;">
                        <span class="dot-live"></span> STREAMING
                    </span>
                </div>
                <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.8rem;">
                    Simulating real-time incoming audience reviews across film screenings:
                </p>
                <div class="stream-card" id="liveStreamFeed">
                    <!-- Populated by dynamic stream animation in JS -->
                </div>
            </div>
        </div>

        <!-- 6. METHODOLOGY GUIDE -->
        <div class="card">
            <h2>🎯 Sentiment Scoring & Rating Calibration</h2>
            <p style="color: var(--text-secondary); margin-bottom: 0.8rem;">
                Audience reactions are evaluated via natural language polarity scoring, mapping contextual sentiment directly to numerical reception:
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 1rem;">
                    <strong style="color: #34d399;">Positive Classification (&ge; +0.05)</strong>
                    <p style="font-size: 0.82rem; color: var(--text-secondary); margin-top: 0.3rem;">
                        High praise, enthusiasm, exceptional craftsmanship (e.g. <em>"Masterpiece!"</em> &rarr; +0.66). Corresponds to 8-10 ratings.
                    </p>
                </div>
                <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 1rem;">
                    <strong style="color: #fbbf24;">Neutral Classification (-0.05 to +0.05)</strong>
                    <p style="font-size: 0.82rem; color: var(--text-secondary); margin-top: 0.3rem;">
                        Balanced, mild, or factual impressions (e.g. <em>"Average film."</em> &rarr; 0.00). Corresponds to 6-7 ratings.
                    </p>
                </div>
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 1rem;">
                    <strong style="color: #f87171;">Negative Classification (&le; -0.05)</strong>
                    <p style="font-size: 0.82rem; color: var(--text-secondary); margin-top: 0.3rem;">
                        Criticism, dissatisfaction, poor pacing (e.g. <em>"Waste of time."</em> &rarr; -0.42). Corresponds to 1-5 ratings.
                    </p>
                </div>
            </div>
        </div>

        <footer>
            <p>© CineMetrics Analytics • Real-Time Movie Intelligence Platform</p>
        </footer>
    </div>

    <!-- Application JavaScript -->
    <script>
        // Parse backend JSON data for 10-year graphs
        const chartsData = {{ charts_data_json | safe }};

        // --- GRAPH 1: 10-Year Rating & Box Office Trend ---
        const ratingsCtx = document.getElementById('ratingsTrendChart').getContext('2d');
        
        const ratingGradient = ratingsCtx.createLinearGradient(0, 0, 0, 300);
        ratingGradient.addColorStop(0, 'rgba(56, 189, 248, 0.4)');
        ratingGradient.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

        const ratingsChart = new Chart(ratingsCtx, {
            type: 'line',
            data: {
                labels: chartsData.yearly.years,
                datasets: [
                    {
                        label: 'Average Movie Rating (1 - 10)',
                        data: chartsData.yearly.ratings,
                        borderColor: '#38bdf8',
                        backgroundColor: ratingGradient,
                        fill: true,
                        tension: 0.35,
                        borderWidth: 3,
                        pointBackgroundColor: '#38bdf8',
                        pointBorderColor: '#ffffff',
                        pointRadius: 5,
                        pointHoverRadius: 8,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Avg Box Office ($ Millions)',
                        data: chartsData.yearly.box_office,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.15)',
                        fill: false,
                        tension: 0.35,
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointBackgroundColor: '#10b981',
                        pointRadius: 4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8', font: { size: 12 } }
                    },
                    tooltip: {
                        backgroundColor: '#1e293b',
                        titleColor: '#f8fafc',
                        bodyColor: '#cbd5e1',
                        borderColor: '#334155',
                        borderWidth: 1,
                        padding: 10
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#1f293d' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        min: 4,
                        max: 10,
                        grid: { color: '#1f293d' },
                        ticks: { color: '#38bdf8' },
                        title: { display: true, text: 'Audience Rating ⭐', color: '#38bdf8' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        min: 0,
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#10b981' },
                        title: { display: true, text: 'Box Office ($M)', color: '#10b981' }
                    }
                }
            }
        });

        // Filter button for Chart 1
        function setChartFilter(mode, btn) {
            document.querySelectorAll('.filter-btn-group .filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            if (mode === 'recent') {
                ratingsChart.data.labels = chartsData.yearly.years.slice(5);
                ratingsChart.data.datasets[0].data = chartsData.yearly.ratings.slice(5);
                ratingsChart.data.datasets[1].data = chartsData.yearly.box_office.slice(5);
            } else {
                ratingsChart.data.labels = chartsData.yearly.years;
                ratingsChart.data.datasets[0].data = chartsData.yearly.ratings;
                ratingsChart.data.datasets[1].data = chartsData.yearly.box_office;
            }
            ratingsChart.update();
        }

        // --- GRAPH 2: 10-Year Sentiment Composition ---
        const sentimentCtx = document.getElementById('sentimentBarChart').getContext('2d');
        const sentimentChart = new Chart(sentimentCtx, {
            type: 'bar',
            data: {
                labels: chartsData.yearly.years,
                datasets: [
                    {
                        label: 'Positive 😊',
                        data: chartsData.yearly.positive,
                        backgroundColor: '#10b981'
                    },
                    {
                        label: 'Neutral 😐',
                        data: chartsData.yearly.neutral,
                        backgroundColor: '#f59e0b'
                    },
                    {
                        label: 'Negative 😞',
                        data: chartsData.yearly.negative,
                        backgroundColor: '#ef4444'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8' } }
                },
                scales: {
                    x: { stacked: true, grid: { color: '#1f293d' }, ticks: { color: '#94a3b8' } },
                    y: { stacked: true, grid: { color: '#1f293d' }, ticks: { color: '#94a3b8' } }
                }
            }
        });

        // --- GRAPH 3: Genre Performance Across the Decade ---
        const genreCtx = document.getElementById('genreChart').getContext('2d');
        const genreChart = new Chart(genreCtx, {
            type: 'bar',
            data: {
                labels: chartsData.genre.genres,
                datasets: [
                    {
                        label: 'Decade Avg Rating ⭐',
                        data: chartsData.genre.ratings,
                        backgroundColor: [
                            'rgba(56, 189, 248, 0.75)',
                            'rgba(168, 85, 247, 0.75)',
                            'rgba(244, 114, 182, 0.75)',
                            'rgba(52, 211, 153, 0.75)',
                            'rgba(251, 191, 36, 0.75)',
                            'rgba(99, 102, 241, 0.75)',
                            'rgba(239, 68, 68, 0.75)',
                            'rgba(14, 165, 233, 0.75)'
                        ],
                        borderRadius: 6
                    }
                ]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { min: 4, max: 10, grid: { color: '#1f293d' }, ticks: { color: '#94a3b8' } },
                    y: { grid: { color: '#1f293d' }, ticks: { color: '#f8fafc' } }
                }
            }
        });

        // --- 10-YEAR TIMELINE ANIMATION (MOVING DATA FEATURE) ---
        const yearsList = chartsData.yearly.years;
        let currentYearIdx = 0;
        let timelineTimer = null;

        const chipsContainer = document.getElementById('yearChips');
        yearsList.forEach((yr, idx) => {
            const btn = document.createElement('button');
            btn.className = 'year-chip' + (idx === 0 ? ' active' : '');
            btn.textContent = yr;
            btn.onclick = () => selectYear(idx);
            chipsContainer.appendChild(btn);
        });

        function selectYear(idx) {
            currentYearIdx = idx;
            document.querySelectorAll('.year-chip').forEach((c, i) => {
                c.classList.toggle('active', i === idx);
            });

            const yr = yearsList[idx];
            const rating = chartsData.yearly.ratings[idx];
            const bo = chartsData.yearly.box_office[idx];
            const pos = chartsData.yearly.positive[idx];
            const neu = chartsData.yearly.neutral[idx];
            const neg = chartsData.yearly.negative[idx];
            const total = pos + neu + neg;
            const posPct = total > 0 ? Math.round((pos / total) * 100) : 0;

            document.getElementById('dispYear').textContent = yr;
            document.getElementById('dispRating').textContent = rating + ' / 10 ⭐';
            document.getElementById('dispBoxOffice').textContent = '$' + bo + 'M';
            document.getElementById('dispSentiment').textContent = `😊 ${posPct}% Positive`;

            // Highlight point on rating chart
            ratingsChart.setActiveElements([{ datasetIndex: 0, index: idx }]);
            ratingsChart.tooltip.setActiveElements([{ datasetIndex: 0, index: idx }], { x: 0, y: 0 });
            ratingsChart.update();
        }

        function toggleTimelinePlay() {
            if (timelineTimer) {
                clearInterval(timelineTimer);
                timelineTimer = null;
                document.getElementById('playIcon').textContent = '▶';
                document.getElementById('playText').textContent = 'Play 10-Year Evolution';
            } else {
                document.getElementById('playIcon').textContent = '⏸';
                document.getElementById('playText').textContent = 'Pause Timeline';
                timelineTimer = setInterval(() => {
                    currentYearIdx = (currentYearIdx + 1) % yearsList.length;
                    selectYear(currentYearIdx);
                }, 3600);
            }
        }

        function resetTimeline() {
            if (timelineTimer) toggleTimelinePlay();
            selectYear(0);
        }

        // Initialize with first year
        selectYear(0);

        // --- MOVING LIVE STREAM SIMULATION (ALIGNED TO REAL ARCHIVE DATA) ---
        const streamItems = [
            { id: 'M2024_01', genre: 'Action', text: 'High-octane car chases, explosive combat, and thrilling martial arts stunts!', rating: '9/10', sentiment: 'Positive', emoji: '😊', color: 'success' },
            { id: 'M2023_01', genre: 'Romance', text: 'Heartwarming romance with tender love and undeniable chemistry.', rating: '9/10', sentiment: 'Positive', emoji: '😊', color: 'success' },
            { id: 'M2018_02', genre: 'Sci-Fi', text: 'Mind-bending sci-fi with spaceships, time travel, and alien worlds.', rating: '10/10', sentiment: 'Positive', emoji: '😊', color: 'success' },
            { id: 'M2017_02', genre: 'Horror', text: 'Terrifying haunted house with eerie jump scares and sinister demons.', rating: '8/10', sentiment: 'Positive', emoji: '😊', color: 'success' },
            { id: 'M2024_03', genre: 'Thriller', text: 'Intense murder mystery with psychological suspense and a shocking twist.', rating: '9/10', sentiment: 'Positive', emoji: '😊', color: 'success' }
        ];

        let streamIdx = 0;
        function updateLiveStream() {
            const feed = document.getElementById('liveStreamFeed');
            const item = streamItems[streamIdx];
            streamIdx = (streamIdx + 1) % streamItems.length;

            const elem = document.createElement('div');
            elem.className = 'stream-item';
            elem.innerHTML = `
                <div class="stream-item-top">
                    <a href="javascript:void(0)" onclick="lookupMovieId('${item.id}')" style="color: #38bdf8; font-weight: 700; text-decoration: underline;">${item.id} • ${item.genre} 🔍</a>
                    <span class="badge-sentiment ${item.color}">${item.emoji} ${item.sentiment}</span>
                </div>
                <div style="font-size: 0.88rem; color: #f1f5f9; margin-bottom: 0.3rem;">"${item.text}"</div>
                <div style="font-size: 0.78rem; color: #94a3b8; display: flex; justify-content: space-between;">
                    <span>⭐ ${item.rating}</span>
                    <span style="color: #10b981;">Incoming Just Now</span>
                </div>
            `;

            feed.insertBefore(elem, feed.firstChild);
            if (feed.children.length > 3) {
                feed.removeChild(feed.lastChild);
            }
        }

        updateLiveStream();
        updateLiveStream();
        setInterval(updateLiveStream, 8000);

        // --- SYNCHRONIZED ARCHIVE FILTERING & ALIGNMENT ---
        function filterArchiveByMovieId(movieId) {
            const cleanId = movieId.toUpperCase().trim();
            const rows = document.querySelectorAll('#reviewsTable tbody tr');
            let matchCount = 0;

            rows.forEach(row => {
                const rowMid = (row.getAttribute('data-movie-id') || row.children[2].innerText).toUpperCase().trim();
                if (rowMid.includes(cleanId)) {
                    row.style.display = '';
                    matchCount++;
                } else {
                    row.style.display = 'none';
                }
            });

            const countDisplay = document.getElementById('searchResultCount');
            if (countDisplay) {
                countDisplay.innerHTML = `✓ Showing all <strong>${matchCount}</strong> reviews for Movie ID <strong style="color: #fbbf24;">${cleanId}</strong> in archive below. <button type="button" onclick="resetArchiveFilter()" class="sample-btn" style="margin-left: 0.5rem; font-size: 0.75rem; border-color: #38bdf8; color: #38bdf8;">Show All Archive (${rows.length})</button>`;
            }
        }

        function resetArchiveFilter() {
            const input = document.getElementById('featureSearchInput');
            if (input) input.value = '';
            const countDisplay = document.getElementById('searchResultCount');
            if (countDisplay) countDisplay.textContent = '';
            
            const rows = document.querySelectorAll('#reviewsTable tbody tr');
            rows.forEach(row => row.style.display = '');

            document.querySelectorAll('.filter-btn-group .filter-btn').forEach(b => b.classList.remove('active'));
            const allBtn = document.querySelector('.filter-btn-group .filter-btn');
            if (allBtn) allBtn.classList.add('active');
        }

        // --- SENTIMENT ANALYZER & MOVIE ID HANDLERS ---
        function setSample(text) {
            document.getElementById('reviewInput').value = text;
            analyzeSentimentAsync(text);
        }

        function lookupMovieId(movieId) {
            const cleanId = movieId.trim().toUpperCase();
            document.getElementById('reviewInput').value = cleanId;
            const searchInput = document.getElementById('featureSearchInput');
            if (searchInput) searchInput.value = cleanId;

            // 1. Fetch complete intelligence report for this movie (genre, rating, consensus, etc.)
            analyzeSentimentAsync(cleanId, false);

            // 2. Align archive table immediately to this exact movie
            filterArchiveByMovieId(cleanId);

            const target = document.getElementById('featureSearchSection');
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        }

        const form = document.getElementById('sentimentForm');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            const text = document.getElementById('reviewInput').value.trim();
            if (text) analyzeSentimentAsync(text);
        });

        function analyzeSentimentAsync(text, syncArchive = true) {
            const btn = document.getElementById('analyzeBtn');
            btn.textContent = 'Analyzing...';
            btn.disabled = true;

            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ review_text: text })
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById('placeholderBox').style.display = 'none';
                const container = document.getElementById('resultContainer');
                container.style.display = 'block';

                if (data.is_movie_id && data.movie_data) {
                    // --- CASE 1: MOVIE ID DEEP-DIVE ANALYSIS ---
                    const m = data.movie_data;
                    container.className = 'result-box ' + (m.dominant_color || 'info');
                    
                    document.getElementById('movieAnalysisView').style.display = 'block';
                    document.getElementById('reviewAnalysisView').style.display = 'none';

                    // Fill Movie ID header & badges
                    document.getElementById('mIdVal').textContent = m.movie_id;
                    document.getElementById('mGenreBadge').textContent = (m.genre_emoji || '🎬') + ' ' + m.genre;
                    document.getElementById('mYearVal').textContent = m.release_year;

                    // Fill Key Metrics
                    document.getElementById('mReviewCount').textContent = m.total_reviews + ' Reviews';
                    document.getElementById('mAvgRating').textContent = m.avg_rating + ' ⭐';
                    document.getElementById('mBoxOffice').textContent = m.box_office_formatted;

                    // Fill Dominant Sentiment & Breakdown
                    const mDom = document.getElementById('mDominantSentiment');
                    mDom.className = 'badge-sentiment ' + m.dominant_color;
                    mDom.textContent = m.dominant_emoji + ' ' + m.dominant_sentiment;

                    document.getElementById('mPosReviews').textContent = `${m.pos_count} Positive (${m.pos_pct}%)`;
                    document.getElementById('mNeuReviews').textContent = `${m.neu_count} Neutral (${m.neu_pct}%)`;
                    document.getElementById('mNegReviews').textContent = `${m.neg_count} Negative (${m.neg_pct}%)`;

                    // Populate Reviews List
                    document.getElementById('mListCount').textContent = m.reviews.length;
                    const reviewsContainer = document.getElementById('mReviewsList');
                    reviewsContainer.innerHTML = '';
                    m.reviews.forEach(r => {
                        const item = document.createElement('div');
                        item.style.cssText = 'background: rgba(15, 23, 42, 0.65); border: 1px solid var(--border-color); border-radius: 6px; padding: 0.5rem 0.65rem; font-size: 0.8rem;';
                        item.innerHTML = `
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                                <span style="color: #94a3b8; font-size: 0.72rem; font-weight: 600;">${r.review_id} • User ${r.user_id}</span>
                                <span class="badge-sentiment ${r.color}" style="font-size: 0.72rem; padding: 0.1rem 0.45rem;">${r.emoji} ${r.rating}⭐ ${r.badge_label || r.sentiment}</span>
                            </div>
                            <div style="color: #f1f5f9; line-height: 1.35;">"${r.review_text}"</div>
                            ${r.is_sarcastic ? '<div style="color: #f87171; font-size: 0.72rem; margin-top: 0.2rem;">🙃 Sarcastic tone detected</div>' : ''}
                        `;
                        reviewsContainer.appendChild(item);
                    });

                    // Synchronize the Archive Table below to this EXACT same movie ID!
                    if (syncArchive) {
                        const searchInput = document.getElementById('featureSearchInput');
                        if (searchInput) searchInput.value = m.movie_id;
                        filterArchiveByMovieId(m.movie_id);
                    }

                } else {
                    // --- CASE 2: REGULAR PROMPT / REVIEW SENTIMENT & GENRE ---
                    container.className = 'result-box ' + data.color;
                    document.getElementById('movieAnalysisView').style.display = 'none';
                    document.getElementById('reviewAnalysisView').style.display = 'block';

                    // Update Genre Classification
                    document.getElementById('resGenreEmoji').textContent = data.genre_emoji || '🎬';
                    document.getElementById('resGenre').textContent = data.genre_detected || 'General';
                    document.getElementById('resGenreConf').textContent = (data.genre_confidence || '90%') + ' Match';
                    const cluesText = (data.genre_clues && data.genre_clues.length > 0) 
                        ? data.genre_clues.map(c => `"${c}"`).join(', ') 
                        : 'Contextual film premise';
                    document.getElementById('resGenreClues').textContent = cluesText;

                    // Update Sentiment
                    document.getElementById('resEmoji').textContent = data.emoji;
                    document.getElementById('resSentiment').textContent = data.badge_label || data.sentiment;
                    document.getElementById('resCompound').textContent = data.compound;
                    document.getElementById('resPos').textContent = data.pos + '%';
                    document.getElementById('resNeu').textContent = data.neu + '%';
                    document.getElementById('resNeg').textContent = data.neg + '%';

                    const sarcasmElem = document.getElementById('sarcasmNotice');
                    if (data.is_sarcastic) {
                        sarcasmElem.style.display = 'block';
                        document.getElementById('sarcasmReasonText').textContent = data.explanation;
                    } else {
                        sarcasmElem.style.display = 'none';
                    }

                    // Check if review matched an existing archive movie
                    const linkedBanner = document.getElementById('matchedArchiveMovieBanner');
                    if (data.matched_movie_id) {
                        linkedBanner.style.display = 'block';
                        document.getElementById('matchedMovieTitle').textContent = data.matched_movie_id;
                        document.getElementById('matchedMovieExtra').textContent = `(${data.matched_movie_genre}, ⭐${data.matched_movie_rating}/10, Year ${data.matched_movie_year})`;
                        document.getElementById('matchedMovieBtn').onclick = () => lookupMovieId(data.matched_movie_id);

                        // Automatically filter table to this matched movie in archive below!
                        if (syncArchive) {
                            filterArchiveByMovieId(data.matched_movie_id);
                        }
                    } else {
                        linkedBanner.style.display = 'none';
                    }
                }
            })
            .catch(err => {
                console.error("Error analyzing sentiment:", err);
            })
            .finally(() => {
                btn.textContent = 'Analyze Sentiment & Detect Genre / Movie ID';
                btn.disabled = false;
            });
        }

        // Quick filter by genre button
        function quickFilterGenre(genre) {
            const input = document.getElementById('featureSearchInput');
            input.value = genre;
            handleFeatureSearch(genre);
            const tbl = document.getElementById('archiveTableSection');
            if (tbl) {
                tbl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }

        // --- FEATURE SEARCH FILTER FUNCTION ---
        function handleFeatureSearch(query) {
            const term = query.toLowerCase().trim();
            const rows = document.querySelectorAll('#reviewsTable tbody tr');
            let matchCount = 0;

            // Check if query is a Movie ID pattern (e.g. M2024_01 or M2023_02)
            const isMovieId = /^m\d{4}(_\d{2})?$/i.test(term) || /^m\d+$/i.test(term);

            rows.forEach(row => {
                const text = row.innerText.toLowerCase();
                const rowMid = (row.getAttribute('data-movie-id') || '').toLowerCase().trim();
                const isMatch = isMovieId ? (rowMid === term) : (!term || text.includes(term));
                if (isMatch) {
                    row.style.display = '';
                    matchCount++;
                } else {
                    row.style.display = 'none';
                }
            });

            const countDisplay = document.getElementById('searchResultCount');
            if (term) {
                countDisplay.innerHTML = `✓ Found <strong>${matchCount}</strong> matching movie reviews in archive below. <button type="button" onclick="resetArchiveFilter()" class="sample-btn" style="margin-left: 0.5rem; font-size: 0.75rem; border-color: #38bdf8; color: #38bdf8;">Show All (${rows.length})</button>`;
            } else {
                countDisplay.textContent = '';
            }

            // If user typed a Movie ID into archive search, automatically align top intelligence report!
            if (isMovieId && matchCount > 0) {
                document.getElementById('reviewInput').value = query.trim().toUpperCase();
                analyzeSentimentAsync(query.trim().toUpperCase(), false);
            }
        }

        // --- ARCHIVE TABLE SENTIMENT FILTER BUTTONS ---
        function filterTable(sentiment, btn) {
            document.querySelectorAll('.filter-btn-group .filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Clear feature search input when clicking filter buttons
            document.getElementById('featureSearchInput').value = '';
            document.getElementById('searchResultCount').textContent = '';

            const rows = document.querySelectorAll('#reviewsTable tbody tr');
            rows.forEach(row => {
                if (sentiment === 'all' || row.getAttribute('data-sentiment') === sentiment) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>

```

---

### 4. `static/style.css`

```css
:root {
    --bg-primary: #0b0f19;
    --bg-secondary: #111827;
    --bg-card: #172033;
    --bg-card-hover: #1e293b;
    --border-color: #283548;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent-blue: #38bdf8;
    --accent-purple: #a855f7;
    --accent-indigo: #6366f1;
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --success: #10b981;
    --success-bg: rgba(16, 185, 129, 0.15);
    --warning: #f59e0b;
    --warning-bg: rgba(245, 158, 11, 0.15);
    --danger: #ef4444;
    --danger-bg: rgba(239, 68, 68, 0.15);
    --radius: 14px;
    --shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.45);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

body {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
    padding-bottom: 4rem;
}

/* Moving Data Ticker Ribbon */
.ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: linear-gradient(90deg, #111827, #1e293b, #111827);
    border-bottom: 1px solid var(--border-color);
    padding: 0.65rem 0;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(10px);
}

.ticker-content {
    display: flex;
    width: max-content;
    animation: ticker 360s linear infinite;
    gap: 1.5rem;
}

.ticker-wrap:hover .ticker-content {
    animation-play-state: paused;
}

@keyframes ticker {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

.ticker-item {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid var(--border-color);
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    white-space: nowrap;
}

.ticker-tag {
    font-weight: 700;
    color: var(--accent-blue);
}

.ticker-rating {
    color: #fbbf24;
    font-weight: 600;
}

.ticker-text {
    color: var(--text-secondary);
    font-style: italic;
}

.ticker-badge {
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
}

.container {
    max-width: 1200px;
    margin: 2rem auto 0;
    padding: 0 1.25rem;
}

/* Header */
header {
    text-align: center;
    margin-bottom: 2.5rem;
}

.pulse-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(16, 185, 129, 0.1);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 0.25rem 0.85rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}

.dot-live {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}

header p {
    color: var(--text-secondary);
    font-size: 1.15rem;
}

/* Stats Grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2.5rem;
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    padding: 1.35rem 1.2rem;
    border-radius: var(--radius);
    text-align: center;
    box-shadow: var(--shadow);
    transition: transform 0.2s;
}

.stat-card:hover {
    transform: translateY(-2px);
    border-color: #3b82f6;
}

.stat-value {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.stat-label {
    color: var(--text-secondary);
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}

.color-pos { color: var(--success); }
.color-neu { color: var(--warning); }
.color-neg { color: var(--danger); }
.color-rate { color: #38bdf8; }
.color-gold { color: #fbbf24; }

/* Timeline Player Card (Moving Data Simulation) */
.timeline-card {
    background: linear-gradient(135deg, #172033 0%, #1e1b4b 100%);
    border: 1px solid #3730a3;
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow);
}

.timeline-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1.25rem;
}

.timeline-header h3 {
    font-size: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.timeline-controls {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.play-btn {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    padding: 0.45rem 1rem;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    transition: all 0.2s;
}

.play-btn:hover {
    filter: brightness(1.15);
    transform: scale(1.02);
}

.year-chips {
    display: flex;
    gap: 0.35rem;
    overflow-x: auto;
    padding-bottom: 0.25rem;
}

.year-chip {
    background: #0f172a;
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    padding: 0.3rem 0.65rem;
    border-radius: 6px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s;
}

.year-chip.active {
    background: var(--accent-indigo);
    color: white;
    border-color: var(--accent-indigo);
    font-weight: 700;
}

.timeline-display {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    background: rgba(11, 15, 25, 0.6);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1.1rem;
    margin-top: 1rem;
}

.timeline-metric-title {
    font-size: 0.78rem;
    color: var(--text-secondary);
    text-transform: uppercase;
}

.timeline-metric-val {
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 0.2rem;
}

/* Charts Grid */
.charts-grid {
    display: grid;
    grid-template-columns: 2fr 1.2fr;
    gap: 1.5rem;
    margin-bottom: 2rem;
}

@media (max-width: 960px) {
    .charts-grid {
        grid-template-columns: 1fr;
    }
}

.card {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 1.75rem;
    box-shadow: var(--shadow);
    margin-bottom: 2rem;
}

.card-header-flex {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.card h2 {
    font-size: 1.35rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.chart-container {
    position: relative;
    width: 100%;
    min-height: 320px;
}

/* Live Stream simulation card */
.stream-card {
    background: linear-gradient(180deg, #172033, #0f172a);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1.2rem;
    margin-top: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.stream-item {
    background: #111827;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 0.85rem;
    transition: all 0.4s ease;
    animation: slideIn 0.4s ease-out;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
}

.stream-item-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
}

/* Sentiment Analyzer Section */
.analyzer-grid {
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    gap: 1.5rem;
}

@media (max-width: 820px) {
    .analyzer-grid {
        grid-template-columns: 1fr;
    }
}

.input-group {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

textarea {
    width: 100%;
    min-height: 110px;
    padding: 0.9rem;
    background-color: #0b0f19;
    border: 1px solid var(--border-color);
    border-radius: 10px;
    color: var(--text-primary);
    font-size: 1rem;
    resize: vertical;
    outline: none;
    transition: border-color 0.2s;
}

textarea:focus {
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
}

.search-box-wrap {
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-color);
    position: relative;
}

.search-input {
    width: 100%;
    padding: 0.75rem 1rem 0.75rem 2.6rem;
    background: #0b0f19;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 0.95rem;
    outline: none;
    transition: border-color 0.2s;
}

.search-input:focus {
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
}

.search-icon-pos {
    position: absolute;
    left: 0.9rem;
    bottom: 1.15rem;
    color: var(--text-secondary);
    pointer-events: none;
    font-size: 1rem;
}

.sample-prompts {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    color: var(--text-secondary);
}

.sample-btn {
    background: #1e293b;
    border: 1px solid var(--border-color);
    color: #93c5fd;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.8rem;
    transition: all 0.2s;
}

.sample-btn:hover {
    background: #334155;
    color: #fff;
    border-color: #60a5fa;
}

.btn-primary {
    align-self: flex-start;
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    font-weight: 600;
    padding: 0.75rem 1.75rem;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.95rem;
    transition: all 0.2s;
}

.btn-primary:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
}

/* Result Box */
.result-box {
    padding: 1.25rem;
    border-radius: 10px;
    animation: fadeIn 0.3s ease-in-out;
}

.result-box.success {
    background-color: var(--success-bg);
    border: 1px solid var(--success);
}

.result-box.warning {
    background-color: var(--warning-bg);
    border: 1px solid var(--warning);
}

.result-box.danger {
    background-color: var(--danger-bg);
    border: 1px solid var(--danger);
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.sentiment-badge {
    font-size: 1.35rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.score-text {
    font-size: 1rem;
    font-weight: 600;
}

.scores-breakdown {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    margin-top: 0.8rem;
    font-size: 0.82rem;
}

.score-item {
    background: rgba(0, 0, 0, 0.3);
    padding: 0.5rem;
    border-radius: 6px;
    text-align: center;
}

.score-item span {
    display: block;
    font-weight: 700;
    margin-top: 0.2rem;
}

/* Table Section */
.table-filter-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.filter-btn-group {
    display: flex;
    gap: 0.4rem;
}

.filter-btn {
    background: #0f172a;
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.82rem;
    transition: all 0.2s;
}

.filter-btn.active, .filter-btn:hover {
    background: var(--primary);
    color: white;
    border-color: var(--primary);
}

.table-wrapper {
    overflow-x: auto;
    max-height: 480px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    text-align: left;
}

th {
    background-color: #0f172a;
    color: var(--text-secondary);
    font-weight: 600;
    padding: 0.85rem 1rem;
    position: sticky;
    top: 0;
    z-index: 1;
    border-bottom: 1px solid var(--border-color);
}

td {
    padding: 0.7rem 1rem;
    border-bottom: 1px solid #1f293d;
    color: var(--text-primary);
}

tr:hover {
    background-color: var(--bg-card-hover);
}

.badge-sentiment {
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}

.badge-sentiment.success {
    background: var(--success-bg);
    color: var(--success);
    border: 1px solid var(--success);
}

.badge-sentiment.warning {
    background: var(--warning-bg);
    color: var(--warning);
    border: 1px solid var(--warning);
}

.badge-sentiment.danger {
    background: var(--danger-bg);
    color: var(--danger);
    border: 1px solid var(--danger);
}

/* Footer */
footer {
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-top: 3.5rem;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-5px); }
    to { opacity: 1; transform: translateY(0); }
}

```

---

### 5. `dataset.csv` (Sample)

The dataset contains 200 movie reviews spanning the decade from **2015 to 2024** across 50 unique movies (4 reviews per movie):

```csv
review_id,movie_id,user_id,rating,review_text,genre,release_year,box_office_earnings
1,M2015_01,U001,9,Heartwarming romance with tender love and undeniable chemistry.,Romance,2015,13540975
2,M2015_01,U002,8,Charming performances and gorgeous Italian scenery.,Romance,2015,13540975
3,M2015_01,U003,7,"Decent romantic watch, though slightly predictable ending.",Romance,2015,13540975
4,M2015_01,U004,5,Waste of time. Disappointing love story that lacks soul.,Romance,2015,13540975
5,M2015_02,U005,9,Whimsical animated adventure with colorful illustrations and charming voice acting.,Animation,2015,21433478
6,M2015_02,U006,8,Fun family entertainment with a heartwarming message.,Animation,2015,21433478
7,M2015_02,U007,8,Stunning CGI visuals and memorable music.,Animation,2015,21433478
8,M2015_02,U008,7,Enjoyable animation for kids and adults alike.,Animation,2015,21433478
9,M2015_03,U009,10,Absolute cinema! Brilliant direction and mind-bending space exploration.,Sci-Fi,2015,27252451
10,M2015_03,U010,9,Visually stunning sci-fi with incredible time travel mechanics.,Sci-Fi,2015,27252451
11,M2015_03,U011,9,"Mind-bending sci-fi with spaceships, time travel, and alien worlds.",Sci-Fi,2015,27252451
12,M2015_03,U012,7,Decent visual effects though slightly complex to follow.,Sci-Fi,2015,27252451
13,M2015_04,U013,9,"High-octane car chases, explosive combat, and thrilling martial arts stunts!",Action,2015,24755144
14,M2015_04,U014,9,Spectacular soundtrack and breathtaking action sequences.,Action,2015,24755144
15,M2015_04,U015,8,Gripping heist storyline and solid pacing throughout.,Action,2015,24755144
16,M2015_04,U016,6,Decent stuntwork but formulaic screenplay.,Action,2015,24755144
17,M2015_05,U017,10,Masterpiece! A timeless classic with unforgettable performances.,Drama,2015,17776778
18,M2015_05,U018,9,Heartwarming drama with memorable characters and deep emotion.,Drama,2015,17776778
19,M2015_05,U019,8,"Good plot, strong performances, and solid pacing.",Drama,2015,17776778
20,M2015_05,U020,7,"Touching story, though somewhat slow in the second act.",Drama,2015,17776778
21,M2016_01,U021,9,"High-octane car chases, explosive combat, and thrilling martial arts stunts!",Action,2016,25213575
22,M2016_01,U022,9,Spectacular soundtrack and breathtaking action sequences.,Action,2016,25213575
23,M2016_01,U023,8,Gripping heist storyline and solid pacing throughout.,Action,2016,25213575
24,M2016_01,U024,6,Decent stuntwork but formulaic screenplay.,Action,2016,25213575
...
```

---

## 🎯 Key Feature Breakdown

### 1. Movie ID Search & Intelligence Deep-Dive
When a user inputs a Movie ID (such as `M2024_01`, `M2023_02`, `M2020_04`):
1. **Format Detection:** The backend detects the pattern `^M\d+(_\d+)?$` and activates `get_movie_analysis(movie_id)`.
2. **Aggregated Film Intelligence:**
   - **Film Genre:** Classified genre with representative icon (e.g. 💥 Action, 🚀 Sci-Fi).
   - **Release Year:** Release year within the 10-year span (2015–2024).
   - **Available Reviews Count:** Aggregated number of audience reviews in the database.
   - **Average Rating:** Star rating aggregated across all reviewers (e.g. 8.5 / 10 ⭐).
   - **Box Office:** Box office revenue formatted in millions (e.g. $35.0M).
   - **Sentiment Consensus:** Breakdown of positive, neutral, and negative reviews with percentages (e.g. 75% Positive, 25% Neutral, 0% Negative).
   - **Review Feed:** Scrollable view of every individual audience review for this movie, including reviewer ID, rating, sentiment tag, and sarcasm flag.
3. **One-Click Quick Access:**
   - Preset buttons under the search prompt (e.g., `M2024_01`, `M2024_02`, `M2024_03`, `M2023_01`, `M2022_04`, `M2021_02`).
   - Clickable Movie IDs directly in the Decade Film Archive table.

### 2. Dual Sentiment & Genre Engine
When a prompt or review text is entered, the engine concurrently evaluates:
- **Sentiment Polarity:** Positivity (+), Neutrality (0), Negativity (-).
- **Genre Classification:** Identifies semantic keywords for 8 cinema genres:
  - 🚀 **Sci-Fi**: space, spaceship, alien, futuristic, time travel, quantum, cyberpunk.
  - 💥 **Action**: explosion, car chase, martial arts, superhero, combat, velocity.
  - 👻 **Horror**: haunted, ghost, demon, jump scare, nightmare, sinister.
  - 🔍 **Thriller**: suspense, mystery, plot twist, detective, interrogation.
  - 😂 **Comedy**: hilarious, laugh, humor, joke, parody, comic timing.
  - ❤️ **Romance**: love story, chemistry, lovers, heartwarming, relationship.
  - 🎭 **Drama**: emotional, heartbreak, tearjerker, grief, struggle, tragedy.
  - 🎨 **Animation**: cartoon, pixar, disney, anime, voice acting, illustrations.

### 3. Sarcasm & Irony Inverter
Lexicon models often misclassify sarcastic reviews with words like *"great"* or *"loved"* as positive. CineMetrics includes an advanced sarcasm pre-processor that detects:
- Ironic openers (*"Oh great..."*, *"Yeah, because..."*)
- Conditional insults (*"Brilliant, if you have no brain cells"*)
- Contrastive dissonance (*"I just loved sitting in the dark for 3 hours being utterly bored"*)
When sarcasm is caught, the compound score is inverted to negative and tagged **Negative (Sarcastic) 🙃**.

### 4. 10-Year Interactive Graphs & Moving Data
- **Feature Search First:** Placed directly at the top of the workspace.
- **Rating & Box Office Spline Chart:** Dual-axis chart tracking 10-year rating changes and box office revenue.
- **Sentiment Stacked Bar Chart:** Year-by-year composition of positive, neutral, and negative reception.
- **Genre Performance Horizontal Bar Chart:** Decade-long average ratings per genre.
- **4x Slower Moving Ribbon Ticker:** Smoothly moving continuous marquee with movie IDs and live reviews (`animation: ticker 360s linear infinite`).
- **Interactive 10-Year Timeline Player:** Automatically steps through years 2015 to 2024.

---

## 📡 API Reference

### 1. `POST /predict` (Review Text Prompt)
Analyzes any movie review or story prompt for sentiment, sarcasm, and genre.

**Request:**
```http
POST /predict HTTP/1.1
Content-Type: application/json

{
  "review_text": "Mind-bending sci-fi with spaceships, time travel, and alien worlds."
}
```

**Response:**
```json
{
  "badge_label": "Positive",
  "color": "success",
  "compound": 0.0,
  "emoji": "😊",
  "explanation": null,
  "genre_clues": ["time travel", "spaceships", "alien", "sci-fi"],
  "genre_confidence": "98%",
  "genre_detected": "Sci-Fi",
  "genre_emoji": "🚀",
  "is_movie_id": false,
  "is_sarcastic": false,
  "neg": 0.0,
  "neu": 100.0,
  "pos": 0.0,
  "sentiment": "Neutral",
  "text": "Mind-bending sci-fi with spaceships, time travel, and alien worlds."
}
```

---

### 2. `POST /predict` (Movie ID Lookup)
Submitting a Movie ID (e.g. `M2024_01`) returns the complete aggregated film analysis.

**Request:**
```http
POST /predict HTTP/1.1
Content-Type: application/json

{
  "review_text": "M2024_01"
}
```

**Response:**
```json
{
  "is_movie_id": true,
  "movie_data": {
    "movie_id": "M2024_01",
    "genre": "Action",
    "genre_emoji": "💥",
    "release_year": 2024,
    "box_office_earnings": 35000000.0,
    "box_office_formatted": "$35.0M",
    "total_reviews": 4,
    "avg_rating": 8.5,
    "pos_count": 3,
    "neu_count": 1,
    "neg_count": 0,
    "pos_pct": 75.0,
    "neu_pct": 25.0,
    "neg_pct": 0.0,
    "dominant_sentiment": "Positive",
    "dominant_emoji": "😊",
    "dominant_color": "success",
    "avg_compound": 0.654,
    "reviews": [
      {
        "review_id": "181",
        "user_id": "U181",
        "rating": 10.0,
        "review_text": "Masterpiece! A timeless classic.",
        "sentiment": "Positive",
        "badge_label": "Positive",
        "emoji": "😊",
        "color": "success",
        "compound": 0.6597,
        "is_sarcastic": false
      }
    ]
  }
}
```

---

### 3. `GET /movie/<movie_id>`
Direct REST endpoint to fetch analytics for any movie ID.

**Request:**
```http
GET /movie/M2023_01 HTTP/1.1
```

**Response:**
```json
{
  "movie_id": "M2023_01",
  "genre": "Romance",
  "genre_emoji": "❤️",
  "release_year": 2023,
  "box_office_formatted": "$29.8M",
  "total_reviews": 4,
  "avg_rating": 7.2,
  "dominant_sentiment": "Positive",
  "dominant_emoji": "😊",
  "pos_pct": 50.0,
  "neu_pct": 25.0,
  "neg_pct": 25.0,
  "reviews": [...]
}
```

---

## 🔧 Troubleshooting & FAQ

- **How do I look up a specific movie?**  
  Simply type any Movie ID (like `M2024_01`, `M2023_02`, etc.) into the main search box and click Analyze, click one of the yellow Movie ID preset buttons, or click any Movie ID link in the Decade Film Archive table below.
- **Port 5000 Already in Use?**  
  In `app.py`, change line 460 from `port=5000` to `port=5050`, then visit `http://127.0.0.1:5050`.
- **NLTK Lexicon Download Issue?**  
  The code has a self-healing fallback that automatically attempts `nltk.download('vader_lexicon')`, and includes a zero-dependency fallback dictionary if NLTK is offline.
- **How to Stop the Server?**  
  Press <kbd>Ctrl</kbd> + <kbd>C</kbd> in your PowerShell window.
