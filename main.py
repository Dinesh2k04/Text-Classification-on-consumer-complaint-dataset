# ...existing code...
from src.data_preprocessing import preprocess_data
from src.model_training import train_models
from src.model_evaluation import evaluate_model
from src.predict import predict_text
import sys
from pathlib import Path
import joblib
import pandas as pd
import os

# Define categories (labels should match your dataset encoding)
class_names = [
    "Credit reporting, repair, or other",
    "Debt collection",
    "Consumer Loan",
    "Mortgage"
]

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
(MODELS_DIR).mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "reports").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "plots").mkdir(parents=True, exist_ok=True)

def run_pipeline(method="tfidf", sample_text=None):
    """
    Run full pipeline for specified method ('tfidf' or 'bert').
    Saves/loads model/vectorizer with suffix _tfidf or _bert.
    """
    method = method.lower()
    if method not in ("tfidf", "bert"):
        raise ValueError("method must be 'tfidf' or 'bert'")

    suffix = f"_{method}"
    print(f"🚀 ComplaintClassifierX Pipeline Starting ({method})...")

    # 1. Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(method=method)

    # 2. Train models
    # print("\n🧑‍💻 Training models...")
    # try:
    #     results = train_models(X_train, y_train, X_test, y_test, class_names, method=method)
    # except TypeError:
    #     results = train_models(X_train, y_train, X_test, y_test, class_names)

    # 3. Paths for best model & vectorizer/tokenizer (use project models dir)
    best_model_path = MODELS_DIR / f"best_model{suffix}.pkl"
    if method == "tfidf":
        vectorizer_path = MODELS_DIR / f"tfidf_vectorizer{suffix}.pkl"
    else:
        vectorizer_path = MODELS_DIR / f"bert_model{suffix}.pkl"

    # 4. Evaluate best model (load model object, no label encoder handling)
    print("\n📊 Evaluating best model...")
    if best_model_path.exists():
        best_model = joblib.load(best_model_path)
        evaluate_model(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            model_name=f"best_model{suffix}",
            plots_dir=str(OUTPUTS_DIR / "plots"),
            reports_dir=str(OUTPUTS_DIR / "reports")
        )
    else:
        print(f"Best model not found at {best_model_path}")

    # 5. Predict new sample
    if sample_text is None:
        sample_text = "I have been charged unfairly by the mortgage company."

    print("\n🔮 Predicting a sample complaint...")
    pred_category = predict_text(
        sample_text,
        model_path=str(best_model_path),
        vectorizer_path=str(vectorizer_path),
        class_names=class_names
    )

    print(f"\n🔮 Sample Complaint: {sample_text}")
    print(f"👉 Predicted Category: {pred_category}")

    return {
        "method": method,
        "best_model_path": str(best_model_path),
        "vectorizer_path": str(vectorizer_path),
        "prediction": pred_category
    }

def evaluate_saved_models(method="tfidf"):
    """
    Load saved LogisticRegression and RandomForest models for given method suffix,
    evaluate both on the test set and save a comparison CSV in outputs/reports.
    (No label-encoder handling; expects models accept y_test strings if trained that way.)
    """
    method = method.lower()
    if method not in ("tfidf", "bert"):
        raise ValueError("method must be 'tfidf' or 'bert'")
    suffix = f"_{method}"

    # get test data
    _, X_test, _, y_test = preprocess_data(method=method)

    model_candidates = {
        "LogisticRegression": MODELS_DIR / f"logisticregression{suffix}.pkl",
        "RandomForest": MODELS_DIR / f"randomforest{suffix}.pkl"
    }

    eval_results = {}
    for name, path in model_candidates.items():
        if not path.exists():
            print(f"Model file not found: {path} — skipping {name}")
            continue
        model = joblib.load(path)
        print(f"\n▶️ Evaluating {name} (file: {path.name})")
        res = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            model_name=f"{name.lower()}{suffix}",
            plots_dir=str(OUTPUTS_DIR / "plots"),
            reports_dir=str(OUTPUTS_DIR / "reports")
        )
        eval_results[name] = res

    # comparison summary
    rows = []
    for name, r in eval_results.items():
        rows.append({
            "model": name,
            "accuracy": r.get("accuracy"),
            "f1_macro": r.get("f1_macro"),
            "json_report": r.get("json_report_path"),
            "csv_report": r.get("csv_report_path"),
            "cm_plot": r.get("cm_plot_path")
        })
    if rows:
        df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
        cmp_path = OUTPUTS_DIR / "reports" / f"comparison{suffix}.csv"
        df.to_csv(cmp_path, index=False)
        print(f"\n✅ Comparison saved to {cmp_path}")
        print(df[["model", "accuracy", "f1_macro"]].to_string(index=False))
    else:
        print("No models evaluated — nothing to compare.")

    return eval_results

if __name__ == "__main__":
    # Usage:
    #  - Run full pipeline: python main.py tfidf
    #  - Run BERT pipeline:  python main.py bert
    #  - Evaluate saved models: python main.py evaluate tfidf
    if len(sys.argv) > 1 and sys.argv[1].lower() == "evaluate":
        method = sys.argv[2] if len(sys.argv) > 2 else "tfidf"
        evaluate_saved_models(method)
    else:
        method = sys.argv[1] if len(sys.argv) > 1 else "tfidf"
        run_pipeline(method)