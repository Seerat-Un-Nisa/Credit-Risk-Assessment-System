import streamlit as st
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# API CONFIGURATION
# ============================================================

API_URL = "https://credit-risk-assessment-system-production.up.railway.app/predict"


# ============================================================
# PAGE TITLE
# ============================================================

st.title("💳 Credit Risk Assessment System")

st.write(
    "Enter borrower information below to assess the probability "
    "of loan default using the trained Logistic Regression model."
)

st.divider()


# ============================================================
# BORROWER INFORMATION
# ============================================================

st.header("Borrower Information")
st.caption("Example borrower information is pre-filled for demonstration. Modify the values as required.")
col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=60000.0,
        step=1000.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=15000.0,
        step=1000.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=700
    )


with col2:

    months_employed = st.number_input(
        "Months Employed",
        min_value=0,
        value=60
    )

    num_credit_lines = st.number_input(
        "Number of Credit Lines",
        min_value=0,
        value=4
    )

    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=8.5,
        step=0.1
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1,
        value=36
    )


with col3:

    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01
    )

    education = st.selectbox(
        "Education",
        [
            "Bachelor's",
            "High School",
            "Master's",
            "PhD"
        ]
    )

    employment_type = st.selectbox(
        "Employment Type",
        [
            "Full-time",
            "Part-time",
            "Self-employed",
            "Unemployed"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Divorced",
            "Married",
            "Single"
        ]
    )


# ============================================================
# ADDITIONAL INFORMATION
# ============================================================

st.header("Additional Loan Information")

col4, col5, col6, col7 = st.columns(4)

with col4:

    has_mortgage = st.selectbox(
        "Has Mortgage",
        ["Yes", "No"]
    )

with col5:

    has_dependents = st.selectbox(
        "Has Dependents",
        ["Yes", "No"]
    )

with col6:

    loan_purpose = st.selectbox(
        "Loan Purpose",
        [
            "Auto",
            "Business",
            "Education",
            "Home",
            "Other"
        ]
    )

with col7:

    has_cosigner = st.selectbox(
        "Has Co-Signer",
        ["Yes", "No"]
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔍 Assess Credit Risk",
    type="primary",
    use_container_width=True
):

    payload = {

        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,

        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner
    }

    try:

        with st.spinner("Assessing credit risk..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            st.success("Credit risk assessment completed successfully.")

            st.divider()

            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            st.header("Risk Assessment Result")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:

                st.metric(
                    "Default Probability",
                    f"{result['default_probability'] * 100:.2f}%"
                )

            with result_col2:

                st.metric(
                    "Risk Score",
                    f"{result['risk_score']:.2f}"
                )

            with result_col3:

                st.metric(
                    "Risk Category",
                    result["risk_category"]
                )

            st.subheader("Decision")

            if result["decision"] == "APPROVED":

                st.success(
                    f"✅ {result['decision']}"
                )

            elif result["decision"] == "MANUAL REVIEW":

                st.warning(
                    f"⚠️ {result['decision']}"
                )

            else:

                st.error(
                    f"❌ {result['decision']}"
                )

        else:

            st.error(
                f"API returned status code {response.status_code}"
            )

            st.code(response.text)

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the FastAPI server. "
            "Could not connect to the Credit Risk Assessment API. Please make sure the Railway API is online."
        )

    except requests.exceptions.Timeout:

        st.error(
            "The API request timed out."
        )

    except Exception as e:

        st.error(
            f"Unexpected error: {str(e)}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Risk Assessment System | "
    "Machine Learning + FastAPI + Streamlit"
)