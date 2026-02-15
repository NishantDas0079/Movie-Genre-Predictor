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
    with open('dashboard_stats.json', 'r') as f:
        stats = json.load(f)

    # 1. Genre frequency bar chart
    genre_counts = stats['genre_counts']
    fig1 = px.bar(x=list(genre_counts.values()), y=list(genre_counts.keys()),
                  orientation='h', title='Genre Frequency in Training Data',
                  labels={'x': 'Count', 'y': 'Genre'},
                  color=list(genre_counts.values()), color_continuous_scale='Viridis')
    plot1 = plot(fig1, output_type='div', include_plotlyjs='cdn')

    # 2. Plot length distribution
    plot_lengths = stats['plot_lengths']
    fig2 = px.histogram(x=plot_lengths, nbins=50,
                        title='Distribution of Plot Lengths (words)',
                        labels={'x': 'Word Count'}, marginal='box')
    plot2 = plot(fig2, output_type='div')

    # 3. F1 scores per genre
    f1_scores = stats['f1_scores']
    genres = list(f1_scores.keys())
    scores = list(f1_scores.values())
    fig3 = px.bar(x=genres, y=scores, title='F1 Score per Genre (Test Set)',
                  labels={'x': 'Genre', 'y': 'F1 Score'},
                  color=scores, color_continuous_scale='Reds')
    plot3 = plot(fig3, output_type='div')

    # 4. Word cloud
    all_text = stats['all_text']
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          colormap='viridis').generate(all_text)
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    wordcloud_img = base64.b64encode(img.getvalue()).decode()

    return render_template('dashboard.html',
                           plot1=plot1,
                           plot2=plot2,
                           plot3=plot3,
                           wordcloud_img=wordcloud_img)


if __name__ == '__main__':
    app.run(debug=True)
