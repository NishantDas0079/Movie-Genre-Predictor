# app.py
from flask import Flask, request, render_template
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import json


# Download NLTK stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the trained model and genre list
model = joblib.load('movie_genre_model.pkl')
top_genres = joblib.load('top_genres.pkl')

app = Flask(__name__)

# ------------------------------------------------------------
# Text cleaning function (must match training preprocessing)
# ------------------------------------------------------------
def clean_text(text):
    """Lowercase, remove non‑letters, remove stopwords."""
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# ------------------------------------------------------------
# Home page – prediction form
# ------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ------------------------------------------------------------
# Result page – shows predicted genres and probabilities
# ------------------------------------------------------------
@app.route('/result', methods=['POST'])
def result():
    plot_text = request.form['plot']
    cleaned = clean_text(plot_text)

    # Predict probabilities (shape: (n_samples, n_classes))
    proba = model.predict_proba([cleaned])[0]

    # Get binary predictions using threshold 0.5
    predictions = (proba >= 0.5).astype(int)
    predicted_genres = [top_genres[i] for i, val in enumerate(predictions) if val == 1]

    # Create list of (genre, probability) for display, sorted by probability
    genre_probs = list(zip(top_genres, proba))
    genre_probs.sort(key=lambda x: x[1], reverse=True)

    return render_template('result.html',
                           plot=plot_text,
                           predicted_genres=predicted_genres,
                           genre_probs=genre_probs)

# ------------------------------------------------------------
# Dashboard – interactive data visualisations
# ------------------------------------------------------------
@app.route('/dashboard')
def dashboard():
    # Load precomputed stats
    with open('genre_counts.json', 'r') as f:
        genre_counts = json.load(f)
    with open('f1_scores.json', 'r') as f:
        f1_scores = json.load(f)
    with open('plot_length_hist.json', 'r') as f:
        hist_data = json.load(f)

    # 1. Genre frequency bar chart
    fig1 = px.bar(x=list(genre_counts.values()), y=list(genre_counts.keys()),
                  orientation='h', title='Genre Frequency in Training Data',
                  labels={'x': 'Count', 'y': 'Genre'},
                  color=list(genre_counts.values()), color_continuous_scale='Viridis')
    plot1 = plot(fig1, output_type='div', include_plotlyjs='cdn')

    # 2. Plot length distribution (from histogram bins)
    bin_edges = hist_data['bin_edges']
    counts = hist_data['counts']
    # Create a bar chart with bins
    fig2 = px.bar(x=bin_edges[:-1], y=counts, 
                  title='Distribution of Plot Lengths (words)',
                  labels={'x': 'Word Count', 'y': 'Frequency'},
                  nbins=50)
    plot2 = plot(fig2, output_type='div')

    # 3. F1 scores per genre
    genres = list(f1_scores.keys())
    scores = list(f1_scores.values())
    fig3 = px.bar(x=genres, y=scores, title='F1 Score per Genre (Test Set)',
                  labels={'x': 'Genre', 'y': 'F1 Score'},
                  color=scores, color_continuous_scale='Reds')
    plot3 = plot(fig3, output_type='div')

    # 4. Word cloud – serve the static image
    return render_template('dashboard.html',
                           plot1=plot1,
                           plot2=plot2,
                           plot3=plot3,
                           wordcloud_img='/static/wordcloud.png')


if __name__ == '__main__':
    app.run(debug=True)
