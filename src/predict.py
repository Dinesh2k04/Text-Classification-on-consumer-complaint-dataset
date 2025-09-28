from .utils import load_model, load_vectorizer

def predict_text(text, model_path, vectorizer_path, class_names):
    """
    Predict the category of a single complaint text
    """
    # Load model & vectorizer
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)

    # Transform text
    X_vec = vectorizer.transform([text])

    # Predict
    pred = model.predict(X_vec)[0]
    return class_names[pred]
