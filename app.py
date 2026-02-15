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
    # Load preprocessed data (or a sample) for visualisations
    df = pd.read_csv('movies_preprocessed.csv')

    # 1. Genre frequency bar chart
    genre_counts = df[top_genres].sum().sort_values(ascending=False)
    fig1 = px.bar(x=genre_counts.values, y=genre_counts.index,
                  orientation='h',
                  title='Genre Frequency in Training Data',
                  labels={'x': 'Count', 'y': 'Genre'},
                  color=genre_counts.values,
                  color_continuous_scale='Viridis')
    plot1 = plot(fig1, output_type='div', include_plotlyjs='cdn')

    # 2. Plot length distribution
    fig2 = px.histogram(df, x='plot_words', nbins=50,
                        title='Distribution of Plot Lengths (words)',
                        labels={'plot_words': 'Word Count'},
                        marginal='box')
    plot2 = plot(fig2, output_type='div')

    # 3. Model performance (F1 scores per genre) – from test set evaluation
    #    These numbers are from your last grid search; you can replace them
    #    with actual computed metrics if you have them.
    f1_scores = [0.51, 0.45, 0.45, 0.48, 0.23, 0.17, 0.21, 0.69,
                 0.21, 0.29, 0.24, 0.17, 0.15, 0.34, 0.20]  # order must match top_genres
    fig3 = px.bar(x=top_genres, y=f1_scores,
                  title='F1 Score per Genre (Test Set)',
                  labels={'x': 'Genre', 'y': 'F1 Score'},
                  color=f1_scores,
                  color_continuous_scale='Reds')
    plot3 = plot(fig3, output_type='div')

    # 4. Word cloud from all cleaned plots
    all_text = ' '.join(df['clean_plot'].dropna())
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          colormap='viridis').generate(all_text)
    # Convert word cloud to PNG and then to base64 for embedding in HTML
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