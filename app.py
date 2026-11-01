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
