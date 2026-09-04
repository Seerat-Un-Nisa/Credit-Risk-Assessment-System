# Credit Risk Assessment System

An end-to-end machine learning application that predicts **loan default risk** from borrower financial and demographic information.

The system combines a trained machine learning model with a **FastAPI prediction API**, **Streamlit dashboard**, **Docker**, and **Railway deployment** to provide an accessible credit-risk assessment workflow.

---

## 🚀 Live Demo

| Service                 | Link                                                                  |
| ----------------------- | --------------------------------------------------------------------- |
| **Streamlit Dashboard** | https://dashboard-production-703e.up.railway.app/                     |
| **FastAPI API**         | https://credit-risk-assessment-system-production.up.railway.app/      |
| **Swagger Docs**        | https://credit-risk-assessment-system-production.up.railway.app/docs  |
| **ReDoc**               | https://credit-risk-assessment-system-production.up.railway.app/redoc |

The dashboard allows users to enter borrower information and receive a:

* Default probability
* Risk score
* Risk category
* Lending decision

---

## 📌 Overview

Credit risk assessment is a common machine learning application in financial services. This project predicts the probability that a borrower will default on a loan and converts that probability into an easy-to-understand risk assessment.

The application accepts borrower attributes such as income, loan amount, credit score, employment history, debt-to-income ratio, loan term, and other demographic and financial information.

The trained model is exposed through a REST API, while the Streamlit dashboard provides a user-friendly interface for making predictions.

---

## ✨ Features

* Exploratory data analysis
* Data preprocessing
* Feature engineering
* Multiple classification models
* Stratified cross-validation
* Hyperparameter tuning
* Model evaluation using multiple metrics
* SHAP-based explainable AI analysis
* Saved ML model and preprocessing pipeline
* FastAPI REST API
* Interactive Streamlit dashboard
* Dockerized backend
* Railway cloud deployment
* Swagger and ReDoc API documentation
* Automated risk categorization and decision output

---

## 🏗️ Architecture

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
│    FastAPI API       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Feature Engineering  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ ColumnTransformer    │
│ Preprocessing        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Logistic Regression  │
│       Model          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Risk Assessment      │
│                      │
│ Probability          │
│ Risk Score           │
│ Risk Category        │
│ Decision             │
└──────────────────────┘
```

---

## 📂 Project Structure

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

> If `api_predicton.png` is a typo in the actual repository, rename it to `api_prediction.png`.

---

# 📊 Dataset

The project uses a credit-risk dataset containing **255,347 borrower records**.

The target variable is `Default`, representing whether a borrower defaulted.

### Target Distribution

| Class       | Percentage |
| ----------- | ---------: |
| Non-default |     88.39% |
| Default     |     11.61% |

Because the dataset is imbalanced, model performance is evaluated using more than accuracy, including **Precision, Recall, F1-score, ROC-AUC, and PR-AUC**.

---

# 🧮 Feature Engineering

The project creates additional features to represent relationships between borrower characteristics and financial risk.

| Feature                | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `MonthlyIncome`        | Annual income converted to monthly income      |
| `EstimatedMonthlyDebt` | Estimated monthly debt based on income and DTI |
| `LoanToIncome`         | Loan amount relative to income                 |
| `CreditToAgeRatio`     | Credit score relative to borrower age          |
| `EmploymentStability`  | Employment duration relative to age            |
| `HighRiskCombo`        | Combination of high DTI and low credit score   |
| `InterestDebtStress`   | Interest rate combined with DTI                |
| `LoanPerCreditLine`    | Loan amount relative to number of credit lines |

---

# 🤖 Machine Learning

Three classification algorithms were evaluated:

* Logistic Regression
* Random Forest
* HistGradientBoosting

The final deployed model is **Logistic Regression**.

The preprocessing pipeline uses a saved `ColumnTransformer` to ensure that prediction data is transformed consistently with the training data.

### Dataset Processing

```text
Training samples: 204,277
Testing samples:   51,070

Original features: 24
Processed features: 39
```

---

# 📈 Model Performance

The baseline models were evaluated on the test set:

| Model                | Accuracy | Precision | Recall |     F1 |    ROC-AUC |
| -------------------- | -------: | --------: | -----: | -----: | ---------: |
| Logistic Regression  |   0.8869 |    0.6168 | 0.0695 | 0.1249 | **0.7617** |
| Random Forest        |   0.8854 |    0.6471 |      — |      — |     0.7416 |
| HistGradientBoosting |   0.8865 |    0.6053 |      — |      — |     0.7572 |

### Cross-Validation

A **5-fold Stratified Cross-Validation** procedure was used to evaluate model stability.

```text
Mean ROC-AUC: 0.7558
ROC-AUC Std:  0.0031
Mean PR-AUC:  0.3264
```

### Hyperparameter Tuning

The Logistic Regression model was tuned using `RandomizedSearchCV` with 3-fold cross-validation and 10 parameter candidates.

Selected parameters:

```text
C = 0.07459
solver = liblinear
class_weight = None
```

The tuned model achieved:

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

**SHAP** was used to analyze feature importance and understand how borrower characteristics influence model predictions.

Important features identified during the analysis included:

* Age
* Interest Rate
* LoanToIncome
* MonthsEmployed
* HasCoSigner
* HasDependents

The project also includes an individual prediction explanation demonstrating how specific borrower characteristics contribute to a prediction.

---

# 🚀 FastAPI

The machine learning model is served through a **FastAPI REST API**.

The API loads:

* Trained Logistic Regression model
* Saved preprocessing pipeline

Incoming borrower data is processed using the same feature engineering and preprocessing logic used during training.

### API Endpoints

| Method | Endpoint      | Description              |
| ------ | ------------- | ------------------------ |
| `GET`  | `/`           | API information          |
| `GET`  | `/health`     | Health status            |
| `GET`  | `/model-info` | Model information        |
| `POST` | `/predict`    | Generate risk prediction |

### Example Prediction

**Request**

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

**Response**

```json
{
  "default_probability": 0.0249,
  "risk_score": 2.49,
  "risk_category": "Low Risk",
  "decision": "APPROVED"
}
```

---

# ⚠️ Risk Classification

The predicted default probability is converted into a risk category and decision.

| Default Probability | Risk Category | Decision      |
| ------------------: | ------------- | ------------- |
|            `< 0.15` | Low Risk      | APPROVED      |
|       `0.15 – 0.35` | Medium Risk   | MANUAL REVIEW |
|            `> 0.35` | High Risk     | REJECTED      |

The risk score is calculated as:

```text
Risk Score = Default Probability × 100
```

For example:

```text
Default Probability = 0.1984
Risk Score          = 19.84
Risk Category       = Medium Risk
Decision            = MANUAL REVIEW
```

---

# 🖥️ Streamlit Dashboard

The project includes an interactive **Streamlit dashboard** for making predictions without directly interacting with the API.

The dashboard allows users to enter borrower information and displays:

* Default probability
* Risk score
* Risk category
* Decision

Possible decisions:

```text
APPROVED
MANUAL REVIEW
REJECTED
```

### Dashboard

https://dashboard-production-703e.up.railway.app/

---

# 🐳 Docker

The FastAPI backend is containerized using Docker.

The container includes:

* FastAPI application
* Trained machine learning model
* Preprocessing pipeline

The project uses **Python 3.12** for the API container.

### Build

```bash
docker build -t credit-risk-api .
```

### Run

```bash
docker run --rm -p 8000:8000 credit-risk-api
```

The local API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# ☁️ Deployment

The application is deployed on **Railway** as separate backend and dashboard services.

### Backend

https://credit-risk-assessment-system-production.up.railway.app/

### Dashboard

https://dashboard-production-703e.up.railway.app/

The deployed Streamlit dashboard communicates with the deployed FastAPI backend.

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Seerat-Un-Nisa/Credit-Risk-Assessment-System.git
cd Credit-Risk-Assessment-System
```

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

### Start FastAPI

```bash
uvicorn api.main:app --reload --port 8000
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

### Start Streamlit

In a separate terminal:

```bash
streamlit run dashboard/app.py
```

The dashboard will normally be available at:

```text
http://localhost:8501
```

Make sure the dashboard is configured to use the correct FastAPI backend URL.

---

# 📓 Notebook

The complete machine learning workflow is available in:

```text
notebook/Credit_Risk_Assessment_System_FINAL.ipynb
```

The notebook contains the detailed:

* Data analysis
* Preprocessing
* Feature engineering
* Model training
* Model comparison
* Cross-validation
* Hyperparameter tuning
* Evaluation
* SHAP analysis
* Model persistence

---

# 📸 Screenshots

The `screenshots/` directory contains examples of the deployed system, including:

* FastAPI Swagger documentation
* API health check
* Model information
* API prediction
* Streamlit dashboard
* Dashboard input form
* Approved prediction
* Manual review prediction
* Rejected prediction

---

# 🛠️ Tech Stack

| Category          | Technologies                           |
| ----------------- | -------------------------------------- |
| Language          | Python                                 |
| Data Science      | Pandas, NumPy                          |
| Machine Learning  | Scikit-learn                           |
| Explainable AI    | SHAP                                   |
| Backend           | FastAPI, Uvicorn, Pydantic             |
| Dashboard         | Streamlit                              |
| HTTP Client       | Requests                               |
| Model Persistence | Joblib                                 |
| Containerization  | Docker                                 |
| Deployment        | Railway                                |
| Development       | Jupyter Notebook, VS Code, Git, GitHub |

---

# 🔐 Limitations & Disclaimer

This project is an **educational machine learning implementation** and is not intended to be used as a production financial decision-making system.

A real-world lending system would require additional:

* Model validation
* Monitoring
* Security controls
* Fairness and bias analysis
* Regulatory compliance
* Domain expert review

The risk thresholds used by this project are **project-defined thresholds** and should not be interpreted as actual banking or regulatory lending policies.

---

# 👤 Author

**Seerat-Un-Nisa**

GitHub:
https://github.com/Seerat-Un-Nisa

---

## 📄 License

This project is intended for **educational and portfolio purposes**.
