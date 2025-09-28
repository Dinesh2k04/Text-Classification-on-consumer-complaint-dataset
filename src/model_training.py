# ...existing code...
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from src.model_evaluation import evaluate_model  # ensure evaluation uses same function
from pathlib import Path
import os
from sklearn.preprocessing import LabelEncoder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_models(X_train, y_train, X_test, y_test, class_names, method="tfidf"):
    """
    Train multiple classifiers and save them using a suffix based on `method`.
    Matches main.py expectations: saves best model as ../models/best_model_{method}.pkl
    and individual models as ../models/{model_name}_{method}.pkl
    """
    method = method.lower()
    if method not in ("tfidf", "bert"):
        raise ValueError("method must be 'tfidf' or 'bert'")

    suffix = f"_{method}"
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=10),
        "RandomForest": RandomForestClassifier(n_estimators=10, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\n🔄 Training {name} ({method})...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        print(f"\n📊 Classification Report for {name} ({method}):")
        print(classification_report(y_test, y_pred, zero_division=0))

        results[name] = {
            "model": model,
            "report": report
        }

        # Save each model with method suffix
        safe_name = name.lower().replace(" ", "_")
        out_path = MODELS_DIR / f'{safe_name}{suffix}.pkl'
        joblib.dump(model, str(out_path))

        # Save evaluation plots/reports per model (passes model object)
        evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            model_name=f"{safe_name}{suffix}"
        )

    # Select best model based on macro F1-score
    best_model_name = max(results, key=lambda x: results[x]['report']['macro avg']['f1-score'])
    best_model = results[best_model_name]['model']
    # Save best model with expected filename used in main.py
    joblib.dump(best_model, str(MODELS_DIR / f'best_model{suffix}.pkl'))

    print(f"\n🏆 Best model ({method}): {best_model_name} "
          f"(F1-macro: {results[best_model_name]['report']['macro avg']['f1-score']:.4f})")

    return results