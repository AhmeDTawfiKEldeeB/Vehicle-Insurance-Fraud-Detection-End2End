from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, fbeta_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import DATA_PATH, DROP_COLUMNS, TARGET_COLUMN, TRAIN_ARTIFACT_PATH
from src.features import engineer_features


def find_best_threshold(y_true: pd.Series, y_prob: pd.Series, beta: float = 2.0) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0

    for threshold in [x / 100 for x in range(20, 81)]:
        y_pred = (y_prob >= threshold).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def train_model(data_path: Path, artifact_path: Path, test_size: float = 0.2) -> dict:
    df = pd.read_csv(data_path)
    df = engineer_features(df)

    y = df[TARGET_COLUMN].astype(str).str.upper().map({"Y": 1, "N": 0})
    X = df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS, errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / max(pos, 1))

    model = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        gamma=1.0,
        min_child_weight=4,
        reg_alpha=0.5,
        reg_lambda=4.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    best_threshold, best_f2 = find_best_threshold(y_test, y_prob, beta=2.0)
    y_pred = (y_prob >= best_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f2": float(best_f2),
        "threshold": float(best_threshold),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
    }

    artifact = {
        "pipeline": pipeline,
        "threshold": best_threshold,
        "metrics": metrics,
        "feature_columns": X_train.columns.tolist(),
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection pipeline.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to CSV data.")
    parser.add_argument("--artifact", default=str(TRAIN_ARTIFACT_PATH), help="Path to save model artifact.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    args = parser.parse_args()

    metrics = train_model(Path(args.data), Path(args.artifact), test_size=args.test_size)

    print("Training complete.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Recall:   {metrics['recall']:.4f}")
    print(f"F2:       {metrics['f2']:.4f}")
    print(f"Threshold:{metrics['threshold']:.2f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
