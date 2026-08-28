from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"


# ============================================================
# LOAD SAVED MODEL AND PREPROCESSOR
# ============================================================

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Credit Risk Assessment API",
    description="REST API for credit default risk prediction.",
    version="1.0.0",
)


# ============================================================
# INPUT DATA SCHEMA
# ============================================================

class LoanApplication(BaseModel):

    Age: float
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: float
    InterestRate: float
    LoanTerm: float
    DTIRatio: float

    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:

    df = data.copy()

    # --------------------------------------------------------
    # Financial Features
    # --------------------------------------------------------

    df["MonthlyIncome"] = df["Income"] / 12

    df["EstimatedMonthlyDebt"] = (
        df["MonthlyIncome"] * df["DTIRatio"]
    )

    df["LoanToIncome"] = (
        df["LoanAmount"] / (df["Income"] + 1)
    )

    # --------------------------------------------------------
    # Risk Indicators
    # --------------------------------------------------------

    df["CreditToAgeRatio"] = (
        df["CreditScore"] / (df["Age"] + 1)
    )

    df["EmploymentStability"] = (
        df["MonthsEmployed"] /
        (df["Age"] * 12 + 1)
    )

    df["HighRiskCombo"] = (
        (df["DTIRatio"] > 0.6) &
        (df["CreditScore"] < 580)
    ).astype(int)

    df["InterestDebtStress"] = (
        df["InterestRate"] * df["DTIRatio"]
    )

    df["LoanPerCreditLine"] = (
        df["LoanAmount"] /
        (df["NumCreditLines"] + 1)
    )

    return df


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Credit Risk Assessment API is running",
        "model": "Logistic Regression",
        "model_features": 39
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model": type(model).__name__,
        "preprocessor": type(preprocessor).__name__,
        "number_of_features": int(model.coef_.shape[1]),
        "classes": model.classes_.tolist()
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(application: LoanApplication):

    # --------------------------------------------------------
    # Convert API request into DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [application.model_dump()]
    )

    # --------------------------------------------------------
    # Recreate training-time engineered features
    # --------------------------------------------------------

    engineered_data = engineer_features(input_data)

    # --------------------------------------------------------
    # Apply the SAVED preprocessing pipeline
    # --------------------------------------------------------

    processed_data = preprocessor.transform(
        engineered_data
    )

    # --------------------------------------------------------
    # Default probability
    # --------------------------------------------------------

    default_probability = float(
        model.predict_proba(processed_data)[0][1]
    )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    risk_score = round(
        default_probability * 100,
        2
    )

    # --------------------------------------------------------
    # Risk category
    # --------------------------------------------------------

    if default_probability < 0.15:

        risk_category = "Low Risk"

    elif default_probability <= 0.35:

        risk_category = "Medium Risk"

    else:

        risk_category = "High Risk"

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if default_probability < 0.15:

        decision = "APPROVED"

    elif default_probability <= 0.35:

        decision = "MANUAL REVIEW"

    else:

        decision = "REJECTED"

    # --------------------------------------------------------
    # API response
    # --------------------------------------------------------

    return {
        "default_probability": round(
            default_probability,
            4
        ),
        "risk_score": risk_score,
        "risk_category": risk_category,
        "decision": decision
    }