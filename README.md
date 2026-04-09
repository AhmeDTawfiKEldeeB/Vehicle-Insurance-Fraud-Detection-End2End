# 🚗 Vehicle Insurance Fraud Detection (End-to-End)

An end-to-end machine learning project for detecting potentially fraudulent insurance claims.

This repository includes:

- 📊 Data processing and feature engineering
- 🤖 Model training and artifact export
- ⚡ FastAPI for real-time prediction
- 🖥️ Streamlit app for interactive use
- 🐳 Docker setup for one-command deployment

## 🖼️ App Preview

![Vehicle Insurance Fraud Detection App](assets/Screenshot%202026-04-09%20123930.png)

---

## ✨ Highlights

- Uses XGBoost for fraud prediction
- Returns fraud probability and risk interpretation
- Human-friendly risk labels:
	- High Risk (Likely Fraud)
	- Medium Risk (Suspicious)
	- Low Risk (Likely Legitimate)

---

## 🧱 Project Structure

```text
.
├── api/
│   └── app.py                     # FastAPI service
├── src/
│   ├── train.py                   # Training pipeline
│   ├── predict.py                 # Inference logic
│   ├── features.py                # Feature engineering
│   ├── config.py                  # Paths and constants
│   └── risk.py                    # Risk-level classification helper
├── streamlit_app.py               # Streamlit UI
├── notebooks/EDA.ipynb            # Exploration + experimentation
├── notebooks/xgboost_full_pipeline.pkl  # Default runtime artifact
├── artifacts/fraud_pipeline.joblib      # Training output artifact
├── Dockerfile.api
├── Dockerfile.streamlit
└── docker-compose.yml
```

---

## ⚙️ Requirements

- Python 3.11+ (recommended)
- pip (or uv)
- Docker Desktop (optional, for containers)

---

## 🚀 Quick Start (Local)

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

or

```bash
uv sync
```

### 2) (Optional) Train a model

```bash
python -m src.train
```

Custom paths:

```bash
python -m src.train --data Data/car_insurance_fraud_dataset.csv --artifact artifacts/fraud_pipeline.joblib
```

### 3) Run API

```bash
uvicorn api.app:app --reload
```

API will be available at:

- http://localhost:8000
- Docs: http://localhost:8000/docs

### 4) Run Streamlit app

```bash
streamlit run streamlit_app.py
```

Streamlit will be available at:

- http://localhost:8501

---

## 🐳 Run with Docker

Build and start everything:

```bash
docker compose up --build
```

Services:

- 🔌 FastAPI: http://localhost:8010
- 📘 API Docs: http://localhost:8010/docs
- 🖥️ Streamlit: http://localhost:8501

Stop services:

```bash
docker compose down
```

---

## 🔮 Prediction API

### Endpoint

- POST /predict

### Request body example

```json
{
	"records": [
		{
			"policy_state": "OH",
			"policy_deductible": 500,
			"policy_annual_premium": 1100,
			"insured_age": 35,
			"insured_sex": "MALE",
			"insured_education_level": "MD",
			"incident_date": "2015-01-25",
			"incident_type": "Single Vehicle Collision",
			"collision_type": "Rear Collision",
			"incident_severity": "Major Damage",
			"incident_state": "SC",
			"incident_hour_of_the_day": 13,
			"number_of_vehicles_involved": 1,
			"bodily_injuries": 0,
			"witnesses": 2,
			"police_report_available": "YES",
			"claim_amount": 7000,
			"total_claim_amount": 12000
		}
	]
}
```

### Response example

```json
{
	"predictions": [
		{
			"risk_level": "Low",
			"risk_label": "Low Risk (Likely Legitimate)",
			"probability_score": 0.36,
			"explanation": "This claim appears legitimate with limited signs of fraud.",
			"fraud_probability": 0.36213627457618713,
			"fraud_prediction": 0
		}
	],
	"threshold": 0.4553002080320843
}
```

---

## 🎯 Fraud Detection Objective

In fraud detection, Recall is prioritized over Accuracy.

Why?

- Missing a fraudulent claim (false negative) is often more costly.
- Extra manual review (false positive) is usually acceptable compared to missed fraud.

---


