from __future__ import annotations

from typing import Any
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import DATA_PATH, DROP_COLUMNS, INFERENCE_ARTIFACT_PATH, TARGET_COLUMN
from src.features import engineer_features


class FraudPredictor:
    def __init__(self, artifact_path: str | None = None) -> None:
        self.artifact_path = artifact_path or str(INFERENCE_ARTIFACT_PATH)

        if not Path(self.artifact_path).exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.artifact_path}."
            )

        artifact: dict[str, Any] = joblib.load(self.artifact_path)

        self.legacy_mode = "model" in artifact and "pipeline" not in artifact
        self.metrics = artifact.get("metrics", {})

        if self.legacy_mode:
            self.model = artifact["model"]
            self.threshold = float(artifact.get("threshold", 0.5))
            self.feature_columns = artifact["features"]
            self.pipeline = None
            self._fit_legacy_preprocessors()
        else:
            self.pipeline = artifact["pipeline"]
            self.threshold = float(artifact["threshold"])
            self.feature_columns = artifact["feature_columns"]
            self.model = None

    def _fit_legacy_preprocessors(self) -> None:
        base_df = pd.read_csv(DATA_PATH)
        base_df = engineer_features(base_df)
        X = base_df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS + ["incident_date"], errors="ignore")

        # Keep only columns expected by the user-saved model.
        X = X[self.feature_columns]

        self.label_encoders: dict[str, LabelEncoder] = {}
        object_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        for col in object_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        self.numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.scaler = StandardScaler()
        X[self.numeric_cols] = self.scaler.fit_transform(X[self.numeric_cols])

    def predict_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        engineered = engineer_features(data)
        prepared = engineered.drop(columns=DROP_COLUMNS + ["incident_date"], errors="ignore")

        # Ensure all training columns exist during inference.
        for col in self.feature_columns:
            if col not in prepared.columns:
                prepared[col] = 0

        prepared = prepared[self.feature_columns]

        if self.legacy_mode:
            for col, le in self.label_encoders.items():
                mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
                prepared[col] = prepared[col].astype(str).map(mapping).fillna(-1).astype(int)

            prepared[self.numeric_cols] = self.scaler.transform(prepared[self.numeric_cols])
            fraud_probability = self.model.predict_proba(prepared)[:, 1]
        else:
            fraud_probability = self.pipeline.predict_proba(prepared)[:, 1]

        fraud_prediction = (fraud_probability >= self.threshold).astype(int)

        return pd.DataFrame(
            {
                "fraud_probability": fraud_probability,
                "fraud_prediction": fraud_prediction,
            }
        )
