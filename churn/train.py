"""Train churn classifiers on the Telco dataset and print real evaluation metrics.

Usage: python -m churn.train [path/to/telco-churn.csv]
"""

import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
DROP = ["customerID"]


def load(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET] + DROP)
    return X, y


def fit_pipeline(X: pd.DataFrame, model) -> Pipeline:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    return pipe


def evaluate(name: str, pipe: Pipeline, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    print(f"{name}: " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
    return metrics


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/telco-churn.csv"
    X, y = load(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"train={len(X_train)} test={len(X_test)} churn_rate={y.mean():.3f}")

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    best_name, best_pipe, best_auc = None, None, -1.0
    for name, model in candidates.items():
        pipe = fit_pipeline(X_train, model).fit(X_train, y_train)
        metrics = evaluate(name, pipe, X_test, y_test)
        if metrics["roc_auc"] > best_auc:
            best_name, best_pipe, best_auc = name, pipe, metrics["roc_auc"]

    joblib.dump({"pipeline": best_pipe, "model_name": best_name, "features": X.columns.tolist()}, "churn-model.joblib")
    print(f"saved churn-model.joblib ({best_name}, roc_auc={best_auc})")


if __name__ == "__main__":
    main()
