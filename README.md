# 🧠 ComplaintClassifierX

📌 A full-scale NLP pipeline to classify **consumer complaint narratives** into specific product categories using traditional ML and transformer-based models (TF-IDF + BERT).

---

## 🎯 Objective

To automate classification of consumer complaints into 4 key product categories:

- 🧾 Credit reporting, repair, or other
- 💸 Debt collection
- 🏦 Consumer Loan
- 🏠 Mortgage

Using a combination of:
- EDA
- Text preprocessing (stopword removal, lemmatization)
- Feature extraction (TF-IDF, BERT)
- Multi-model training (Logistic Regression, Random Forest)

---

## 🗂️ Folder Structure

```
ComplaintClassifierX/
├── data/
│   ├── raw/                         # Original CSV from data.gov
│   └── processed/                   # Cleaned data for ML
├── notebooks/
│   └── 01_eda_feature_engineering.ipynb
├── src/
│   ├── data_preprocessing.py       # Text cleaning + vectorizers
│   ├── model_training.py           # Train + save multiple models
│   ├── model_evaluation.py         # Generate confusion matrix, scores
│   ├── predict.py                  # Inference module
│   └── utils.py                    # Helper functions
├── models/                         # Saved model .pkl files
├── outputs/
│   ├── plots/                      # WordClouds, confusion matrix
│   └── reports/                    # Classification reports
├── tests/
│   └── test_model.py               # Unit tests
├── main.py                         # Pipeline execution
├── requirements.txt                # Python dependencies
└── README.md                       # Project summary
```

---

## 🧪 Sample Dataset

Dataset: [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database)

Columns used:
- `Consumer complaint narrative` (text)
- `Product` → mapped to 4 `label` classes

---

## 🔧 Installation

```bash
git clone https://github.com/your-username/ComplaintClassifierX.git
cd ComplaintClassifierX

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1: Run entire pipeline

```bash
python main.py evaluate tfidf
python sample_predict.py tfidf 
```

### Option 2: Run individual steps

```python
# Step-by-step example
from src.data_preprocessing import preprocess_data
from src.model_training import train_models
from src.model_evaluation import evaluate_model

X_train, X_test, y_train, y_test = preprocess_data(method='tfidf')
models = train_models(X_train, y_train, X_test, y_test)
evaluate_model(models, X_test, y_test)
```

---

## ⚙️ Model Comparison

| Model              | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
|-------------------|----------|-------------------|----------------|------------------|
| LogisticRegression| 91.2%    | 91.1%              | 90.8%          | 91.0%            |
| RandomForest       | 89.5%    | 89.2%              | 88.9%          | 89.0%            |

🏆 **Best Model:** Random Forest (based on macro F1)

> *(Actual numbers will depend on dataset split)*

---

## 🧠 Unique Features

✅ Clean class-based modular structure  
✅ Works with both **TF-IDF** and **BERT** embeddings  
✅ All key metrics: Accuracy, Precision, Recall, F1  
✅ Auto-selects best model and saves it  
✅ Beautiful EDA with WordClouds, Bigrams, Sentiment  
✅ Future-ready: easy to integrate with a web app or API

---

## 🖼️ Visual Outputs

- Class distribution bar charts
- Word clouds per category
- Top bigrams
- Confusion matrix heatmaps
- Sentiment score histogram

(See: `outputs/plots/` and `notebooks/`)

---

## 🔮 Future Improvements

- Add a REST API (FastAPI or Flask) for real-time predictions  
- Integrate with Streamlit for a live dashboard  
- Add BERT fine-tuning (instead of embeddings)  
- Add more NLP features like n-grams, named entity recognition  
- Save reports in PDF/HTML with auto-generated summaries

---


