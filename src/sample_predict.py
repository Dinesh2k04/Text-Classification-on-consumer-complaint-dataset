# ...existing code...
import sys
from pathlib import Path
import joblib
import json
import csv
from datetime import datetime
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
# 10-sentence sample text for prediction
SAMPLE_TEXT = (
    "I was charged an extra fee on my credit report that I do not recognize. "
    "The company refuses to explain why the payment was taken. "
    "I have emailed support multiple times with no reply. "
    "My loan interest rate increased unexpectedly after refinancing. "
    "I received threatening calls from a debt collector. "
    "The mortgage company lost my payment and now claims I'm late. "
    "My credit score dropped after an incorrect report was filed. "
    "I was denied a consumer loan without a clear reason. "
    "Unauthorized accounts were opened in my name and I want them removed. "
    "I need help getting a refund for a service I never signed up for."
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_obj(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return joblib.load(path)

def prepare_features(text: str, method: str):
    method = method.lower()
    if method == "tfidf":
        vec_path = MODELS_DIR / f"tfidf_vectorizer.pkl"
        vectorizer = load_obj(vec_path)
        return vectorizer.transform([text]), vec_path
    elif method == "bert":
        bert_path = MODELS_DIR / f"bert_model_{method}.pkl"
        bert_model = load_obj(bert_path)
        emb = bert_model.encode([text])
        return emb, bert_path
    else:
        raise ValueError("method must be 'tfidf' or 'bert'")

def safe_json(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_prediction_report(model_name, method, prediction, proba, sample_text):
    fname_base = f"{model_name.lower()}_{method}_prediction"
    json_path = REPORTS_DIR / f"{fname_base}.json"
    csv_path = REPORTS_DIR / f"predictions_{method}.csv"

    payload = {
        "model": model_name,
        "method": method,
        "prediction": safe_json(prediction),
        "probability": safe_json(proba),
        "sample_text": sample_text
    }

    # write json
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(payload, jf, indent=2, default=safe_json)

    # append to CSV (create header if missing)
    header = ["timestamp", "model", "method", "prediction", "probability", "sample_text"]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "model": model_name,
            "method": method,
            "prediction": safe_json(prediction),
            "probability": safe_json(proba),
            "sample_text": sample_text
        })

    return str(json_path), str(csv_path)

def predict_on_models(text: str, method: str = "tfidf"):
    method = method.lower()
    model_files = {
        "LogisticRegression": MODELS_DIR / f"logisticregression_{method}.pkl",
        "RandomForest": MODELS_DIR / f"randomforest_{method}.pkl",
    }

    X_vec, used_model_path = prepare_features(text, method)

    results = {}
    for name, path in model_files.items():
        if not path.exists():
            results[name] = {"error": f"model file not found: {path.name}"}
            continue
        model = load_obj(path)
        try:
            pred = model.predict(X_vec)[0]
        except Exception as e:
            results[name] = {"error": f"prediction failed: {e}"}
            continue

        # try probability if available
        proba = None
        try:
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_vec)
                # convert to list of floats (take max class probability)
                proba = float(np.max(p))
            elif hasattr(model, "decision_function"):
                df = model.decision_function(X_vec)
                # if multiclass returns array; take max score
                if isinstance(df, np.ndarray):
                    proba = float(np.max(df))
                else:
                    proba = safe_json(df)
        except Exception:
            proba = None

        results[name] = {
            "prediction": safe_json(pred),
            "probability": proba,
            "model_file": path.name
        }

        # save per-model report to outputs/reports
        save_prediction_report(name, method, pred, proba, text)

    return results

if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "tfidf"
    
    nltk.download('punkt')

    sentences = sent_tokenize(SAMPLE_TEXT)
    print(f"\n📄 Total sentences: {len(sentences)} — Method: {method}\n")

    # CSV file for combined results
    combined_csv = REPORTS_DIR / f"sentence_predictions_{method}.csv"
    header = ["sentence", "rf_prediction", "logistic_prediction", "rf_probability", "logistic_probability"]
    write_header = not combined_csv.exists()

    with open(combined_csv, "a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=header)
        if write_header:
            writer.writeheader()

        for idx, sentence in enumerate(sentences, 1):
            print(f"\n🧾 Sentence {idx}: {sentence}")
            try:
                res = predict_on_models(sentence, method=method)
            except FileNotFoundError as e:
                print(f"❌ {e}")
                continue

            rf_pred, rf_proba = None, None
            log_pred, log_proba = None, None

            if res.get("RandomForest") and "error" not in res["RandomForest"]:
                rf_pred = res["RandomForest"]["prediction"]
                rf_proba = res["RandomForest"]["probability"]
                print(f"  - RandomForest: ✅ {rf_pred} (prob: {rf_proba})")
            else:
                print(f"  - RandomForest: ERROR → {res.get('RandomForest', {}).get('error', 'missing')}")

            if res.get("LogisticRegression") and "error" not in res["LogisticRegression"]:
                log_pred = res["LogisticRegression"]["prediction"]
                log_proba = res["LogisticRegression"]["probability"]
                print(f"  - LogisticRegression: ✅ {log_pred} (prob: {log_proba})")
            else:
                print(f"  - LogisticRegression: ERROR → {res.get('LogisticRegression', {}).get('error', 'missing')}")

            # Write row into CSV
            writer.writerow({
                "sentence": sentence,
                "rf_prediction": rf_pred,
                "logistic_prediction": log_pred,
                "rf_probability": rf_proba,
                "logistic_probability": log_proba
            })

    print(f"\n✅ All results saved to {combined_csv}")
