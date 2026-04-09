from __future__ import annotations


def classify_risk(probability: float) -> dict[str, str | float]:
    score = round(float(probability), 2)

    if probability > 0.6:
        return {
            "risk_level": "High",
            "risk_label": "High Risk (Likely Fraud)",
            "probability_score": score,
            "explanation": "This claim shows strong fraud indicators and should be reviewed immediately.",
        }

    if probability >= 0.4:
        return {
            "risk_level": "Medium",
            "risk_label": "Medium Risk (Suspicious)",
            "probability_score": score,
            "explanation": "This claim has suspicious patterns and should be checked before final approval.",
        }

    return {
        "risk_level": "Low",
        "risk_label": "Low Risk (Likely Legitimate)",
        "probability_score": score,
        "explanation": "This claim appears legitimate with limited signs of fraud.",
    }
