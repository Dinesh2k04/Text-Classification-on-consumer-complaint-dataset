# src/data_preprocessing.py

import pandas as pd
import numpy as np
import nltk
import string
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import os
# Global variables
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE_DEFAULT = PROJECT_ROOT / 'data' / 'processed' / 'cleaned_complaints.csv'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
# ====================
# Text Cleaning
# ====================
def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)

# ====================
# Load & Preprocess Dataset
# ====================
def load_and_clean_data(path=None):
    # use project-root absolute default
    path = Path(path) if path else DATA_FILE_DEFAULT
    if not path.exists():
        # raise helpful error showing where we looked
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df['clean_text'] = df['Consumer complaint narrative'].apply(clean_text)
    return df

# ====================
# TF-IDF Vectorization
# ====================
def tfidf_vectorizer(df, model_path=None):
    # default to project models dir
    model_path = Path(model_path) if model_path else MODELS_DIR / 'tfidf_vectorizer.pkl'
    # ensure parent exists
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        print(f"Loading existing TF-IDF vectorizer from {model_path} ...")
        vectorizer = joblib.load(model_path)
        X = vectorizer.transform(df['clean_text'])
    else:
        print(f"Creating and fitting new TF-IDF vectorizer and saving to {model_path} ...")
        vectorizer = TfidfVectorizer(max_features=3000)
        X = vectorizer.fit_transform(df['clean_text'])
        joblib.dump(vectorizer, str(model_path))
    
    return X

# ====================
# BERT Embeddings
# ====================
def bert_embedding(df, save_model=True, model_path=None):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['clean_text'].tolist(), show_progress_bar=True)
    
    if save_model:
        model_path = Path(model_path) if model_path else MODELS_DIR / 'bert_model.pkl'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, str(model_path))
    
    return embeddings

# ====================
# Train/Test Split
# ====================
def preprocess_data(method='tfidf', data_path=None):
    df = load_and_clean_data(data_path)
    y = df['label']

    if method == 'tfidf':
        X = tfidf_vectorizer(df)
    elif method == 'bert':
        X = bert_embedding(df)
    else:
        raise ValueError("Invalid method. Choose either 'tfidf' or 'bert'")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test
