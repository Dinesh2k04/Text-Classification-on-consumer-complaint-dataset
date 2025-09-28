import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def evaluate_model(model, X_test, y_test, class_names, model_name="best_model",
                   plots_dir="../outputs/plots", reports_dir="../outputs/reports"):
    """
    Evaluate a trained model using Accuracy, F1-score, Classification Report, and Confusion Matrix.
    Saves reports and plots in the respective folders.
    """
    # Ensure directories exist
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n✅ Model Evaluation Results")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (macro): {f1:.4f}\n")

    # Classification report dict
    report_dict = classification_report(y_test, y_pred, target_names=class_names,
                                        output_dict=True, zero_division=0)

    # Print classification report
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Save JSON report
    json_path = os.path.join(reports_dir, f"{model_name.lower()}_classification.json")
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=4)

    # Save CSV report
    df_report = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(reports_dir, f"{model_name.lower()}_classification.csv")
    df_report.to_csv(csv_path, index=True)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()

    cm_path = os.path.join(plots_dir, f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "report": report_dict,
        "json_report_path": json_path,
        "csv_report_path": csv_path,
        "cm_plot_path": cm_path
    }
