from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "car_insurance_fraud_dataset.csv"
INFERENCE_ARTIFACT_PATH = PROJECT_ROOT / "notebooks" / "xgboost_full_pipeline.pkl"
TRAIN_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "fraud_pipeline.joblib"

TARGET_COLUMN = "fraud_reported"
DROP_COLUMNS = [
    "policy_id",
    "incident_city",
    "insured_occupation",
    "insured_hobbies",
    "authorities_contacted",
]
