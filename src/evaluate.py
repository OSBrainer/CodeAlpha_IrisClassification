import argparse
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main(args):
    pipeline = joblib.load(args.model_path)
    df_test  = pd.read_csv(args.test_data)
    X_test   = df_test.drop(columns=["species"])
    y_true   = df_test["species"]

    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_true, preds)
    print(f"Accuracy: {acc:.3f}\n")
    print("Classification Report:")
    print(classification_report(y_true, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to .joblib model")
    parser.add_argument("--test-data",  required=True, help="Path to test CSV")
    args = parser.parse_args()
    main(args)
