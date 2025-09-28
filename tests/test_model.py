import os
import pytest
import joblib
from src.data_preprocessing import preprocess_data
from src.model_training import train_models
from src.evaluation import evaluate_model
from src.utils import load_model, load_vectorizer

# Define categories
class_names = [
    "Credit reporting, repair, or other",
    "Debt collection",
    "Consumer Loan",
    "Mortgage"
]

def test_training_and_saving():
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(method="tfidf")

    # Train models
    results = train_models(X_train, y_train, X_test, y_test, class_names)

    # Check that each model file exists
    for model_name in results.keys():
        model_path = f"../models/{model_name.lower()}.pkl"
        assert os.path.exists(model_path), f"{model_name} model file not saved!"

    # Check best model exists
    best_model_path = "../models/best_model.pkl"
    assert os.path.exists(best_model_path), "Best model file not saved!"

def test_model_loading():
    best_model_path = "../models/best_model.pkl"
    vectorizer_path = "../models/tfidf_vectorizer.pkl"

    # Load model and vectorizer
    model = load_model(best_model_path)
    vectorizer = load_vectorizer(vectorizer_path)

    assert model is not None, "Failed to load best model"
    assert vectorizer is not None, "Failed to load vectorizer"

def test_evaluation_output():
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(method="tfidf")

    # Load best model
    best_model_path = "../models/best_model.pkl"
    model = load_model(best_model_path)

    # Evaluate
    results = evaluate_model(model, X_test, y_test, class_names, model_name="test_model")

    # Check keys in results
    assert "accuracy" in results
    assert "f1_macro" in results
    assert "confusion_matrix" in results
    assert os.path.exists(results["json_report_path"])
    assert os.path.exists(results["csv_report_path"])
    assert os.path.exists(results["cm_plot_path"])
