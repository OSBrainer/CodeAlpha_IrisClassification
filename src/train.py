import argparse
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def build_pipeline() -> Pipeline:
    """Builds a sklearn Pipeline with scaling + logistic regression."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(multi_class="auto", solver="lbfgs", max_iter=200))
    ])

def main(args):
    df = pd.read_csv(args.input)
    X = df.drop(columns=["species"])
    y = df["species"]

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    joblib.dump(pipeline, args.output)
    print(f"Saved model to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to train CSV")
    parser.add_argument("--output", required=True, help="Where to save the model")
    args = parser.parse_args()
    main(args)
