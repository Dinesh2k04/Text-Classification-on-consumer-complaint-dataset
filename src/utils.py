import os
import joblib

def save_model(model, model_name, models_dir="../models"):
    """
    Save a trained model to the models directory
    """
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{model_name.lower()}.pkl")
    joblib.dump(model, path)
    return path

def load_model(model_path):
    """
    Load a saved model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    return model

def save_vectorizer(vectorizer, vectorizer_name="tfidf_vectorizer", models_dir="../models"):
    """
    Save TF-IDF or BERT vectorizer
    """
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{vectorizer_name}.pkl")
    joblib.dump(vectorizer, path)
    return path

def load_vectorizer(vectorizer_path):
    """
    Load TF-IDF or BERT vectorizer
    """
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer
