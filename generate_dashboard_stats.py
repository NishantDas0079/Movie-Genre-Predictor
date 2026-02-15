import pandas as pd
import json
import joblib
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load preprocessed data
df = pd.read_csv('movies_preprocessed.csv')

# Load model and genre list
model = joblib.load('movie_genre_model.pkl')
top_genres = joblib.load('top_genres.pkl')

# 1. Genre counts (small)
genre_counts = df[top_genres].sum().to_dict()
with open('genre_counts.json', 'w') as f:
    json.dump(genre_counts, f)

# 2. F1 scores (small, from your report)
f1_scores = {
    'unknown': 0.51, 'drama': 0.45, 'comedy': 0.45, 'horror': 0.48,
    'action': 0.23, 'thriller': 0.17, 'romance': 0.21, 'western': 0.69,
    'crime': 0.21, 'adventure': 0.29, 'musical': 0.24, 'crime drama': 0.17,
    'romantic comedy': 0.15, 'science fiction': 0.34, 'film noir': 0.20
}
with open('f1_scores.json', 'w') as f:
    json.dump(f1_scores, f)

# 3. Plot length histogram (binned) – much smaller than full list
plot_lengths = df['plot_words']
hist, bin_edges = np.histogram(plot_lengths, bins=50)
hist_data = {
    'counts': hist.tolist(),
    'bin_edges': bin_edges.tolist()
}
with open('plot_length_hist.json', 'w') as f:
    json.dump(hist_data, f)

# 4. Word cloud – generate and save as PNG
all_text = ' '.join(df['clean_plot'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_text)
wordcloud.to_file('wordcloud.png')  # saves directly

print("✅ Lightweight dashboard files generated.")
