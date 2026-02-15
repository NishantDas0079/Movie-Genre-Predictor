# Phase 4.1: Add class_weight='balanced' to Logistic Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import hamming_loss, f1_score, classification_report
import joblib
import numpy as np

# 1. Load the preprocessed data
df = pd.read_csv('movies_preprocessed.csv')
print("Data shape:", df.shape)

# 2. Define non‑genre columns to exclude
non_genre_cols = ['Plot', 'clean_plot', 'plot_words', 'Release Year']  # add any others
# Automatically select genre columns (all int columns not in non_genre_cols)
top_genres = [col for col in df.columns if col not in non_genre_cols and df[col].dtype in ['int64', 'int']]
print(f"Found {len(top_genres)} genre columns: {top_genres}")

# 3. Prepare features and labels
X = df['clean_plot']
y = df[top_genres].values  # Convert to numpy array (important!)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create pipeline with class_weight='balanced'
pipeline_weighted = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')),
    ('clf', OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')))
])

# 6. Train
print("Training model with class_weight='balanced'...")
pipeline_weighted.fit(X_train, y_train)

# 7. Evaluate
y_pred_weighted = pipeline_weighted.predict(X_test)

print("\n" + "="*60)
print("Results with class_weight='balanced':")
print("Hamming loss:", hamming_loss(y_test, y_pred_weighted))
print("Micro F1:", f1_score(y_test, y_pred_weighted, average='micro'))
print("Macro F1:", f1_score(y_test, y_pred_weighted, average='macro'))
print("\nClassification report (per genre):")
print(classification_report(y_test, y_pred_weighted, target_names=top_genres, zero_division=0))

# 8. (Optional) Save the model
# joblib.dump(pipeline_weighted, 'movie_genre_model_weighted.pkl')
# joblib.dump(top_genres, 'top_genres.pkl')

from sklearn.model_selection import GridSearchCV

# Define the pipeline (same as before)
pipeline_tune = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')))
])

# Parameter grid
param_grid = {
    'tfidf__max_features': [3000, 5000, 7000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__estimator__C': [0.1, 1, 10]
}

# Use micro F1 as scoring metric (common for multi‑label)
grid = GridSearchCV(pipeline_tune, param_grid, cv=2, scoring='f1_micro', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best cross-validation F1 (micro):", grid.best_score_)

# Evaluate on test set
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)

print("Test set performance of best model:")
print("Hamming loss:", hamming_loss(y_test, y_pred_best))
print("Micro F1:", f1_score(y_test, y_pred_best, average='micro'))
print("Macro F1:", f1_score(y_test, y_pred_best, average='macro'))
print("\nClassification report:\n", classification_report(y_test, y_pred_best, target_names=top_genres, zero_division=0))

import joblib

# Save the best model from grid search
joblib.dump(best_model, 'movie_genre_model.pkl')
joblib.dump(top_genres, 'top_genres.pkl')
print("Model and genre list saved.")