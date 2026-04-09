from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DATA_PATH
from src.predict import FraudPredictor
from src.risk import classify_risk


@st.cache_resource
def load_predictor() -> FraudPredictor | None:
    try:
        return FraudPredictor()
    except FileNotFoundError:
        return None


def get_defaults() -> dict:
    if Path(DATA_PATH).exists():
        df = pd.read_csv(DATA_PATH)
        return df.drop(columns=["fraud_reported"], errors="ignore").iloc[0].to_dict()
    return {}


st.set_page_config(page_title="Fraud Detection", page_icon="🚗", layout="wide")
st.title("Vehicle Insurance Fraud Detection")
st.caption("End-to-end demo: input claim details and get fraud risk instantly.")

predictor = load_predictor()
defaults = get_defaults()

if predictor is None:
    st.error("Model artifact not found. Run `python -m src.train` first.")
    st.stop()

with st.form("claim_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        policy_state = st.text_input("Policy State", value=str(defaults.get("policy_state", "OH")))
        policy_deductible = st.number_input("Policy Deductible", value=float(defaults.get("policy_deductible", 500)))
        insured_age = st.number_input("Insured Age", min_value=16, max_value=100, value=int(defaults.get("insured_age", 35)))
        insured_sex = st.selectbox("Insured Sex", ["MALE", "FEMALE"], index=0)
        insured_education_level = st.text_input("Education Level", value=str(defaults.get("insured_education_level", "MD")))

    with col2:
        incident_date = st.text_input("Incident Date (YYYY-MM-DD)", value=str(defaults.get("incident_date", "2015-01-25")))
        incident_type = st.text_input("Incident Type", value=str(defaults.get("incident_type", "Single Vehicle Collision")))
        collision_type = st.text_input("Collision Type", value=str(defaults.get("collision_type", "Rear Collision")))
        incident_severity = st.text_input("Incident Severity", value=str(defaults.get("incident_severity", "Major Damage")))
        incident_state = st.text_input("Incident State", value=str(defaults.get("incident_state", "SC")))

    with col3:
        incident_hour_of_day = st.number_input("Incident Hour", min_value=0, max_value=23, value=int(defaults.get("incident_hour_of_the_day", 13)))
        number_of_vehicles_involved = st.number_input("Vehicles Involved", min_value=1, max_value=10, value=int(defaults.get("number_of_vehicles_involved", 1)))
        bodily_injuries = st.number_input("Bodily Injuries", min_value=0, max_value=10, value=int(defaults.get("bodily_injuries", 0)))
        witnesses = st.number_input("Witnesses", min_value=0, max_value=10, value=int(defaults.get("witnesses", 2)))
        police_report_available = st.selectbox("Police Report Available", ["YES", "NO"], index=0)

    policy_annual_premium = st.number_input("Policy Annual Premium", min_value=0.0, value=float(defaults.get("policy_annual_premium", 1100.0)))
    claim_amount = st.number_input("Claim Amount", min_value=0.0, value=float(defaults.get("claim_amount", 7000.0)))
    total_claim_amount = st.number_input("Total Claim Amount", min_value=0.0, value=float(defaults.get("total_claim_amount", 12000.0)))

    submitted = st.form_submit_button("Predict Fraud Risk")

if submitted:
    payload = {
        "policy_state": policy_state,
        "policy_deductible": policy_deductible,
        "policy_annual_premium": policy_annual_premium,
        "insured_age": insured_age,
        "insured_sex": insured_sex,
        "insured_education_level": insured_education_level,
        "incident_date": incident_date,
        "incident_type": incident_type,
        "collision_type": collision_type,
        "incident_severity": incident_severity,
        "incident_state": incident_state,
        "incident_hour_of_the_day": incident_hour_of_day,
        "number_of_vehicles_involved": number_of_vehicles_involved,
        "bodily_injuries": bodily_injuries,
        "witnesses": witnesses,
        "police_report_available": police_report_available,
        "claim_amount": claim_amount,
        "total_claim_amount": total_claim_amount,
    }

    pred_df = predictor.predict_dataframe(pd.DataFrame([payload]))
    prob = float(pred_df.loc[0, "fraud_probability"])
    pred = int(pred_df.loc[0, "fraud_prediction"])
    risk = classify_risk(prob)

    st.subheader("Prediction Result")
    st.metric("Risk Level", str(risk["risk_label"]))
    st.metric("Fraud Probability", f"{float(risk['probability_score']):.2f}")
    st.write(f"Explanation: {risk['explanation']}")
    st.caption(f"Model class output: {'Fraud' if pred == 1 else 'Not Fraud'}")
    st.caption(f"Decision threshold: {predictor.threshold:.2f}")
