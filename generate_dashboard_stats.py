# generate_dashboard_stats.py
import pandas as pd
import json
import joblib

# Load your preprocessed data (adjust path if needed)
df = pd.read_csv('movies_preprocessed.csv')  # you have this locally, right?

# Load the model and genre list
model = joblib.load('movie_genre_model.pkl')
top_genres = joblib.load('top_genres.pkl')

# Compute genre counts
genre_counts = df[top_genres].sum().to_dict()

# Compute plot length distribution (list of word counts)
plot_lengths = df['plot_words'].tolist() if 'plot_words' in df.columns else []

# Get all cleaned text for word cloud
all_text = ' '.join(df['clean_plot'].dropna()) if 'clean_plot' in df.columns else ""

# Optionally, compute F1 scores (you can add them manually from your earlier report)
f1_scores = {
    'unknown': 0.51, 'drama': 0.45, 'comedy': 0.45, 'horror': 0.48,
    'action': 0.23, 'thriller': 0.17, 'romance': 0.21, 'western': 0.69,
    'crime': 0.21, 'adventure': 0.29, 'musical': 0.24, 'crime drama': 0.17,
    'romantic comedy': 0.15, 'science fiction': 0.34, 'film noir': 0.20
}

# Save everything to JSON
stats = {
    'genre_counts': genre_counts,
    'plot_lengths': plot_lengths,
    'all_text': all_text,
    'f1_scores': f1_scores
}

with open('dashboard_stats.json', 'w') as f:
    json.dump(stats, f)

print("Dashboard stats saved to dashboard_stats.json")