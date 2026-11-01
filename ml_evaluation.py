"""
ml_evaluation.py
Machine Learning Evaluation Module for CineMetrics.
Demonstrates:
1. Supervised Learning: Support Vector Machine (SVM)
2. Unsupervised Learning: K-Means Clustering
Evaluation Parameters: Precision, Recall, and F1-Score ONLY.
"""

import csv
import os
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score

# Step 1: Load Dataset
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
texts, labels = [], []

with open(dataset_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(row['review_text'])
        rating = int(row['rating'])
        # Cinema rating threshold: >= 8 Positive, 6-7 Neutral, <= 5 Negative
        if rating >= 8:
            labels.append('Positive')
        elif rating in (6, 7):
            labels.append('Neutral')
        else:
            labels.append('Negative')

print("=" * 65)
print(f"CINEMETRICS MACHINE LEARNING EVALUATION (Dataset: {len(texts)} reviews)")
print("=" * 65)
print("Class Distribution:", Counter(labels))
print("-" * 65)

# Step 2: Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# 75% Train, 25% Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

classes = ['Positive', 'Neutral', 'Negative']

# -----------------------------------------------------------------
# 1. SUPERVISED MACHINE LEARNING: Support Vector Machine (SVM)
# -----------------------------------------------------------------
print("\n[1] SUPERVISED MACHINE LEARNING: Support Vector Machine (SVM)")
svm_classifier = LinearSVC(random_state=42)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

print(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 52)
for cls in classes:
    p = precision_score(y_test, y_pred_svm, labels=[cls], average='macro', zero_division=0)
    r = recall_score(y_test, y_pred_svm, labels=[cls], average='macro', zero_division=0)
    f = f1_score(y_test, y_pred_svm, labels=[cls], average='macro', zero_division=0)
    print(f"{cls:<14} | {p:<10.2f} | {r:<10.2f} | {f:<10.2f}")

macro_p_svm = precision_score(y_test, y_pred_svm, average='macro', zero_division=0)
macro_r_svm = recall_score(y_test, y_pred_svm, average='macro', zero_division=0)
macro_f1_svm = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)
print("-" * 52)
print(f"{'Overall (Macro)':<14} | {macro_p_svm:<10.2f} | {macro_r_svm:<10.2f} | {macro_f1_svm:<10.2f}")

# -----------------------------------------------------------------
# 2. UNSUPERVISED MACHINE LEARNING: K-Means Clustering (K=3)
# -----------------------------------------------------------------
print("\n[2] UNSUPERVISED MACHINE LEARNING: K-Means Clustering (K=3)")
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_model.fit(X)
cluster_labels = kmeans_model.labels_

# Map cluster IDs to dominant sentiment classes
mapped_preds = np.empty_like(cluster_labels, dtype=object)
for i in range(3):
    mask = (cluster_labels == i)
    if np.sum(mask) > 0:
        majority_label = Counter(y[mask]).most_common(1)[0][0]
        mapped_preds[mask] = majority_label
    else:
        mapped_preds[mask] = 'Neutral'

print(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 52)
for cls in classes:
    p = precision_score(y, mapped_preds, labels=[cls], average='macro', zero_division=0)
    r = recall_score(y, mapped_preds, labels=[cls], average='macro', zero_division=0)
    f = f1_score(y, mapped_preds, labels=[cls], average='macro', zero_division=0)
    print(f"{cls:<14} | {p:<10.2f} | {r:<10.2f} | {f:<10.2f}")

macro_p_km = precision_score(y, mapped_preds, average='macro', zero_division=0)
macro_r_km = recall_score(y, mapped_preds, average='macro', zero_division=0)
macro_f1_km = f1_score(y, mapped_preds, average='macro', zero_division=0)
print("-" * 52)
print(f"{'Overall (Macro)':<14} | {macro_p_km:<10.2f} | {macro_r_km:<10.2f} | {macro_f1_km:<10.2f}")
print("=" * 65)
