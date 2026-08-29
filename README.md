# Credit Risk Assessment System

An end-to-end machine learning system for assessing loan default risk using borrower financial and demographic information. The project includes data preprocessing, feature engineering, machine learning model development, model evaluation and tuning, explainable AI analysis, a FastAPI prediction service, a Streamlit dashboard, Docker containerization, and cloud deployment.

---

## 🚀 Live Demo

### Streamlit Dashboard

**Credit Risk Assessment Dashboard:**
https://dashboard-production-703e.up.railway.app/

Use the dashboard to enter borrower information and receive:

* Default probability
* Risk score
* Risk category
* Approval decision

### FastAPI Backend

**API Base URL:**
https://credit-risk-assessment-system-production.up.railway.app/

### API Documentation

**Swagger UI:**
https://credit-risk-assessment-system-production.up.railway.app/docs

**ReDoc:**
https://credit-risk-assessment-system-production.up.railway.app/redoc

### API Endpoints

**Root:**
https://credit-risk-assessment-system-production.up.railway.app/

**Health Check:**
https://credit-risk-assessment-system-production.up.railway.app/health

**Model Information:**
https://credit-risk-assessment-system-production.up.railway.app/model-info

**Prediction:**
https://credit-risk-assessment-system-production.up.railway.app/predict

The `/predict` endpoint accepts a `POST` request containing borrower information and returns the predicted default probability, risk score, risk category, and decision.

---

## 📌 Project Overview

Credit risk assessment is an important application of machine learning in financial services. The objective of this project is to develop a system that estimates the probability that a borrower will default on a loan.

The system takes borrower characteristics such as:

* Age
* Income
* Loan amount
* Credit score
* Employment duration
* Number of credit lines
* Interest rate
* Loan term
* Debt-to-income ratio
* Education
* Employment type
* Marital status
* Mortgage status
* Dependents
* Loan purpose
* Co-signer status

and produces a risk assessment.

The final system provides both a machine learning prediction and a user-friendly interface for interacting with the prediction API.

---

## 🎯 Objectives

The main objectives of the project are:

1. Perform exploratory data analysis on a large credit-risk dataset.
2. Clean and preprocess the dataset.
3. Engineer meaningful financial and borrower-risk features.
4. Train multiple machine learning classification models.
5. Compare model performance using appropriate classification metrics.
6. Perform stratified cross-validation.
7. Tune the selected model using hyperparameter optimization.
8. Use explainable AI techniques to understand model predictions.
9. Save the trained model and preprocessing pipeline.
10. Build a REST API using FastAPI.
11. Build an interactive dashboard using Streamlit.
12. Containerize the API using Docker.
13. Deploy the backend and dashboard to the cloud.
14. Provide a complete reproducible project structure.

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Borrower Inputs    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    └──────────┬───────────┘
                               │ HTTP POST
                               ▼
                    ┌──────────────────────┐
                    │    FastAPI REST API  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Saved Preprocessor   │
                    │ ColumnTransformer    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Logistic Regression  │
                    │       Model          │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │       Risk Assessment           │
              │                                 │
              │ Default Probability             │
              │ Risk Score                       │
              │ Risk Category                    │
              │ Decision                         │
              └─────────────────────────────────┘
```

---

# 📂 Project Structure

```text
Credit-Risk-Assessment-System/
│
├── api/
│   ├── main.py
│   └── requirements.txt
│
├── dashboard/
│   └── app.py
│
├── models/
│   ├── logistic_regression_model.joblib
│   └── preprocessor.joblib
│
├── notebook/
│   └── Credit_Risk_Assessment_System_FINAL.ipynb
│
├── screenshots/
│   ├── api_health.png
│   ├── api_model_info.png
│   ├── api_predicton.png
│   ├── api_swagger.png
│   ├── dashboard_approved.png
│   ├── dashboard_inputs.png
│   ├── dashboard_manual_review.png
│   ├── dashboard_rejected.png
│   └── streamlit_dashboard.png
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The project uses a large-scale credit-risk dataset containing **255,347 borrower records**.

The original dataset contains borrower information and a binary `Default` target indicating whether a borrower defaulted.

The target distribution is approximately:

* **Non-default:** 88.39%
* **Default:** 11.61%

Because the target is imbalanced, evaluation focuses not only on accuracy but also on precision, recall, F1-score, ROC-AUC, and PR-AUC.

---

# 🧮 Feature Engineering

Additional features were created to capture relationships between borrower characteristics and financial risk.

The engineered features include:

| Feature                | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `MonthlyIncome`        | Annual income converted to monthly income        |
| `EstimatedMonthlyDebt` | Estimated monthly debt based on income and DTI   |
| `LoanToIncome`         | Loan amount relative to income                   |
| `CreditToAgeRatio`     | Credit score relative to borrower age            |
| `EmploymentStability`  | Employment duration relative to age              |
| `HighRiskCombo`        | Combined high-DTI and low-credit-score indicator |
| `InterestDebtStress`   | Interest rate combined with DTI                  |
| `LoanPerCreditLine`    | Loan amount relative to number of credit lines   |

These features were incorporated into the preprocessing and modeling pipeline.

---

# ⚙️ Data Preprocessing

The preprocessing pipeline handles numerical and categorical features separately.

The project uses a saved `ColumnTransformer` preprocessing pipeline.

After preprocessing:

```text
Training samples: 204,277
Testing samples:   51,070

Original features: 24
Processed features: 39
```

The preprocessing pipeline also verifies that the processed training and testing data contain no missing values.

The fitted preprocessing pipeline is saved as:

```text
models/preprocessor.joblib
```

---

# 🤖 Machine Learning Models

Three classification models were evaluated:

1. Logistic Regression
2. Random Forest
3. HistGradientBoosting

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

The final deployed prediction service uses the trained **Logistic Regression** model.

---

# 📈 Model Evaluation

The baseline model comparison showed the following performance on the test set:

| Model                | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| -------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression  |   0.8869 |    0.6168 | 0.0695 | 0.1249 |  0.7617 |
| Random Forest        |   0.8854 |    0.6471 |      — |      — |  0.7416 |
| HistGradientBoosting |   0.8865 |    0.6053 |      — |      — |  0.7572 |

The project also uses PR-AUC because the default class is substantially smaller than the non-default class.

---

# 🔄 Cross-Validation

A **5-fold Stratified Cross-Validation** procedure was used to evaluate model stability while preserving the target-class distribution across folds.

For Logistic Regression, the cross-validation results included:

```text
Mean ROC-AUC: 0.7558
ROC-AUC Std:  0.0031
Mean PR-AUC:  0.3264
```

Stratification is important because the target variable is imbalanced.

---

# 🎛️ Hyperparameter Tuning

Hyperparameter tuning was performed using `RandomizedSearchCV` with:

```text
3-fold cross-validation
10 parameter candidates
```

The tuned Logistic Regression model selected parameters approximately equivalent to:

```text
C = 0.07459
solver = liblinear
class_weight = None
```

The best cross-validation ROC-AUC was approximately:

```text
0.7559
```

The tuned model achieved approximately:

```text
Accuracy:  0.8869
Precision: 0.6150
Recall:    0.0690
F1:        0.1240
ROC-AUC:   0.7617
PR-AUC:    0.3366
```

---

# 🔍 Explainable AI

SHAP was used during the machine learning analysis to investigate which features influenced model predictions.

Important features identified during the analysis included:

* Age
* Interest Rate
* LoanToIncome
* MonthsEmployed
* HasCoSigner
* HasDependents

The project also includes an individual borrower explanation example showing how specific borrower characteristics influence an individual prediction.

---

# 💾 Saved Machine Learning Artifacts

The final trained artifacts are stored in the `models/` directory.

### Logistic Regression Model

```text
models/logistic_regression_model.joblib
```

### Preprocessing Pipeline

```text
models/preprocessor.joblib
```

The saved preprocessing pipeline is essential because the API must transform new borrower data using the same preprocessing procedure used during model training.

---

# 🚀 FastAPI Backend

The REST API is implemented in:

```text
api/main.py
```

The API loads:

* The trained Logistic Regression model
* The saved preprocessing pipeline

It then recreates the required engineered features for incoming borrower data before generating a prediction.

---

## API Endpoints

### GET `/`

Returns basic information confirming that the API is running.

### GET `/health`

Returns the health status of the service.

Example:

```json
{
  "status": "healthy",
  "service": "Credit Risk Assessment API"
}
```

### GET `/model-info`

Returns information about the deployed model and preprocessing pipeline.

Example:

```json
{
  "model": "LogisticRegression",
  "preprocessor": "ColumnTransformer",
  "number_of_features": 39,
  "classes": [0, 1]
}
```

### POST `/predict`

Accepts borrower information and returns a risk assessment.

Example response:

```json
{
  "default_probability": 0.0249,
  "risk_score": 2.49,
  "risk_category": "Low Risk",
  "decision": "APPROVED"
}
```

---

# 📋 Prediction Input

The prediction API accepts the following fields:

```json
{
  "Age": 35,
  "Income": 60000,
  "LoanAmount": 15000,
  "CreditScore": 700,
  "MonthsEmployed": 60,
  "NumCreditLines": 4,
  "InterestRate": 8.5,
  "LoanTerm": 36,
  "DTIRatio": 0.30,
  "Education": "Bachelor's",
  "EmploymentType": "Full-time",
  "MaritalStatus": "Married",
  "HasMortgage": "Yes",
  "HasDependents": "Yes",
  "LoanPurpose": "Home",
  "HasCoSigner": "Yes"
}
```

---

# ⚠️ Risk Classification

The API converts the predicted default probability into a risk category.

| Default Probability | Risk Category | Decision      |
| ------------------: | ------------- | ------------- |
|            `< 0.15` | Low Risk      | APPROVED      |
|       `0.15 – 0.35` | Medium Risk   | MANUAL REVIEW |
|            `> 0.35` | High Risk     | REJECTED      |

The risk score is represented on a 0–100 scale:

```text
Risk Score = Default Probability × 100
```

For example:

```text
Default Probability = 0.1984
Risk Score          = 19.84
Risk Category       = Medium Risk
```

The risk score and probability therefore contain the same underlying information but are presented on different scales for usability.

---

# 🖥️ Streamlit Dashboard

The interactive dashboard is implemented in:

```text
dashboard/app.py
```

The dashboard allows users to enter borrower information through a graphical interface.

The dashboard communicates with the deployed FastAPI backend and displays:

* Default Probability
* Risk Score
* Risk Category
* Decision

Possible decisions are:

```text
APPROVED
MANUAL REVIEW
REJECTED
```

### Dashboard URL

https://dashboard-production-703e.up.railway.app/

---

# 🐳 Docker

The FastAPI application is containerized using Docker.

The project includes:

```text
Dockerfile
.dockerignore
```

The Docker image uses Python 3.12 and installs the API dependencies defined in:

```text
api/requirements.txt
```

The container includes:

* FastAPI application
* Saved machine learning model
* Saved preprocessing pipeline

The API runs using Uvicorn on port `8000`.

### Build Docker Image

```bash
docker build -t credit-risk-api .
```

### Run Container

```bash
docker run --rm -p 8000:8000 credit-risk-api
```

The local API can then be accessed at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# ☁️ Deployment

The project has been deployed using Railway.

The backend and Streamlit dashboard are deployed as separate services.

### Backend

```text
https://credit-risk-assessment-system-production.up.railway.app/
```

### Dashboard

```text
https://dashboard-production-703e.up.railway.app/
```

The Streamlit dashboard communicates with the deployed FastAPI backend rather than the local development server.

---

# 🧪 Testing

The API was tested through the FastAPI Swagger interface and direct HTTP requests.

The following functionality was verified:

* API startup
* Root endpoint
* Health endpoint
* Model information endpoint
* Prediction endpoint
* Dockerized API
* Model loading
* Preprocessor loading
* Prediction generation
* Risk classification
* Approval decision
* Manual review decision
* Rejection decision
* Streamlit dashboard
* Dashboard-to-API communication
* Deployed backend
* Deployed dashboard

Screenshots documenting the testing and dashboard functionality are included in:

```text
screenshots/
```

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Scikit-learn
* SHAP

### Machine Learning

* Logistic Regression
* Random Forest
* HistGradientBoosting
* RandomizedSearchCV
* Stratified Cross-Validation

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend / Dashboard

* Streamlit
* Requests

### Model Persistence

* Joblib

### Containerization

* Docker

### Deployment

* Railway

### Development

* Jupyter Notebook
* Visual Studio Code
* Git
* GitHub

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Seerat-Un-Nisa/Credit-Risk-Assessment-System.git
```

Navigate into the project:

```bash
cd Credit-Risk-Assessment-System
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the FastAPI Backend Locally

From the project root:

```bash
uvicorn api.main:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

# ▶️ Run the Streamlit Dashboard Locally

Start the API first.

Then run:

```bash
streamlit run dashboard/app.py
```

The dashboard will normally become available at:

```text
http://localhost:8501
```

The dashboard must be configured to communicate with the appropriate FastAPI backend URL.

---

# 📓 Notebook

The complete machine learning workflow is available in:

```text
notebook/Credit_Risk_Assessment_System_FINAL.ipynb
```

The notebook contains the project workflow including:

* Dataset loading
* Exploratory data analysis
* Data preprocessing
* Feature engineering
* Feature verification
* Model training
* Model comparison
* Cross-validation
* Hyperparameter tuning
* Model evaluation
* SHAP analysis
* Model persistence

---

# 📸 Screenshots

The repository contains screenshots demonstrating:

* FastAPI Swagger documentation
* API health endpoint
* API model information
* API prediction
* Streamlit dashboard
* Dashboard input form
* Approved prediction
* Manual review prediction
* Rejected prediction

Screenshots are available in:

```text
screenshots/
```

---

# 🔐 Important Notes

This project is an educational machine learning implementation and should not be treated as a production financial decision-making system without additional validation, monitoring, security controls, fairness analysis, regulatory review, and domain expert oversight.

The risk thresholds used by the API are project-defined thresholds and should not be interpreted as actual banking or regulatory lending policies.

---

# 👤 Author

**Seerat-Un-Nisa**

GitHub:

https://github.com/Seerat-Un-Nisa

---

# 📄 License

This project is intended for educational and portfolio purposes.
