# 🎬 Cinema Genius – Movie Genre Predictor

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://movie-genre-predictor-bz5n.onrender.com)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-2.3.3-lightgrey)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Cinema Genius** is a machine learning web application that predicts movie genres from a plot description. It uses a multi‑label classification model trained on the Wikipedia Movie Plots dataset. The app features a sleek, cinematic UI and an interactive dashboard for data exploration.

🌐 **Live Demo**: [https://movie-genre-predictor-bz5n.onrender.com](https://movie-genre-predictor-bz5n.onrender.com)

*(Note: The free tier may spin down after inactivity – the first request might take a few seconds.)*

---

## ✨ Features

- 🎯 **Multi‑label genre prediction** – predicts one or more of 15 genres (Drama, Comedy, Horror, Action, etc.).
- 🖥️ **Beautiful, responsive UI** – dark theme with gold accents, inspired by cinema.
- 🚀 **One‑click sample plots** – test with plots from *The Matrix*, *Titanic*, *Inception*, and more.
- 📊 **Interactive dashboard** – explore genre frequencies, plot length distribution, model performance, and a word cloud.
- 📡 **REST API endpoint** – integrate predictions into your own applications.
- 📱 **Mobile‑friendly** – works seamlessly on all devices.

---

## 🛠️ Tech Stack

| Component          | Technology                                                                 |
|--------------------|----------------------------------------------------------------------------|
| **Backend**        | Python, Flask, Gunicorn                                                    |
| **ML / NLP**       | scikit‑learn, imbalanced‑learn, NLTK, joblib                               |
| **Data Handling**  | pandas, numpy                                                              |
| **Visualisation**  | Plotly, WordCloud, matplotlib                                              |
| **Frontend**       | HTML, CSS, JavaScript                                                      |
| **Deployment**     | Render (free tier)                                                         |

---

## 📈 Model Performance

The model is a **Random Forest** classifier trained with **SMOTE** to handle class imbalance. It was optimised using grid search. Performance on the test set (20% holdout) is:

| Metric        | Value |
|---------------|-------|
| Micro F1      | 0.39  |
| Macro F1      | 0.32  |
| Hamming Loss  | 0.093 |

**Per‑genre F1 scores:**

| Genre            | F1 Score |
|------------------|----------|
| Western          | 0.69     |
| Unknown          | 0.51     |
| Drama            | 0.45     |
| Comedy           | 0.45     |
| Horror           | 0.48     |
| Science Fiction  | 0.34     |
| Adventure        | 0.29     |
| Action           | 0.23     |
| Musical          | 0.24     |
| Romance          | 0.21     |
| Crime            | 0.21     |
| Thriller         | 0.17     |
| Film Noir        | 0.20     |
| Crime Drama      | 0.17     |
| Romantic Comedy  | 0.15     |

*(See the interactive dashboard for visualisations of these metrics.)*

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/NishantDas0079/Movie-Genre-Predictor.git
cd Movie-Genre-Predictor
```

# 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

# 3. Install dependencies
```bash
pip install -r requirements.txt
```

# 4. Download NLTK stopwords (if not already cached)
```python
python -c "import nltk; nltk.download('stopwords')"
```

# 5. Run the Flask App
```bash
python app.py
```

# 6. Open your browser
Go to http://127.0.0.1:5000 and start predicting!

# 📡 API Usage
You can also use the model programmatically via the /predict endpoint.

Endpoint: `https://movie-genre-predictor-bz5n.onrender.com/predict`
Method: POST
Content‑Type: application/json


# 🧠 What I Learned
Building a multi‑label classification pipeline with scikit‑learn.

Handling class imbalance using SMOTE.

Text preprocessing with NLTK (stopwords, tokenization).

Hyperparameter tuning with GridSearchCV.

Creating an interactive dashboard with Plotly and WordCloud.

Deploying a Flask app on Render and troubleshooting version mismatches.

Managing Git history to exclude large files (.gitignore, rewriting commits).

# 🎯 Future Improvements
Add more genres (expand from 15 to 20+).

Incorporate deep learning (e.g., fine‑tuning BERT for text classification).

Allow users to upload a text file.

Add user feedback to improve the model over time.

