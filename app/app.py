import streamlit as st
import requests


import requests
import streamlit as st

# PAGE CONFIG
st.set_page_config(
    page_title="ChurnIQ | Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"

# CUSTOM CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #081120 0%, #0c1730 100%);
    color: #edf2ff;
}

[data-testid="stSidebar"] {
    background: #0a1326;
    border-right: 1px solid #1d3154;
}

[data-testid="stSidebar"] h1 {
    color: #a5b4fc;
}

h1, h2, h3 {
    color: #edf2ff !important;
}

p, label, .stMarkdown {
    color: #c5d2e8;
}

.hero {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(115deg, #142d59, #30235f 70%, #182447);
    border: 1px solid #344c86;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 34px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    margin-bottom: 0;
    color: #c5d5f5;
}

.panel {
    background: #101e34;
    border: 1px solid #263c5c;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}

.metric-card {
    background: #12223c;
    border: 1px solid #2b4266;
    border-radius: 14px;
    padding: 18px;
    min-height: 115px;
}

.metric-label {
    color: #9fb2d1;
    font-size: 13px;
}

.metric-value {
    color: #f3f6ff;
    font-size: 27px;
    font-weight: 800;
    margin-top: 7px;
}

.small-note {
    color: #94a8c8;
    font-size: 12px;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #3678ff, #8655ef);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 48px;
    transition: 0.2s ease;
}

div.stButton > button[kind="primary"]:hover {
    border: 1px solid #a9baff;
    filter: brightness(1.12);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #172943;
    border-color: #354d70;
}

[data-testid="stForm"] {
    background: #101e34;
    border: 1px solid #263c5c;
    border-radius: 16px;
    padding: 22px;
}

hr {
    border-color: #263c5c;
}

[data-testid="stMetric"] {
    background: #12223c;
    border: 1px solid #2b4266;
    border-radius: 12px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)


# SIDEBAR
with st.sidebar:
    st.markdown("# 📊 Customer Churn Prediction System")
    st.caption("CUSTOMER RETENTION INTELLIGENCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Predict Churn", "Model Information", "About"],
        label_visibility="collapsed"
    )

    st.divider()

    try:
        health = requests.get(HEALTH_URL, timeout=3)
        api_online = (
            health.ok and health.json().get("model_loaded", False)
        )
    except requests.exceptions.RequestException:
        api_online = False

    if api_online:
        st.success("API connected")
    else:
        st.error("API offline")

    st.markdown("---")
    st.caption("ML Model")
    st.markdown("**Random Forest Classifier**")
    st.caption("FastAPI · Streamlit · Scikit-learn")
    st.caption("Version 1.0.0")


# REUSABLE FUNCTIONS
def metric_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="small-note">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_prediction(result, customer):
    prediction = result["prediction"]
    probability = result["churn_probability"]
    risk = (
        "High" if probability >= 0.65
        else "Medium" if probability >= 0.35
        else "Low"
    )

    st.divider()
    st.subheader("Prediction Overview")

    a, b, c = st.columns(3)

    with a:
        metric_card(
            "PREDICTED CLASS",
            "Churn" if prediction == 1 else "No Churn",
            "Model classification"
        )

    with b:
        metric_card(
            "CHURN PROBABILITY",
            f"{probability:.2%}",
            "Estimated model probability"
        )

    with c:
        metric_card(
            "RISK INDICATOR",
            risk,
            "Based on probability bands"
        )

    st.markdown("### Probability Analysis")
    st.progress(max(0.0, min(1.0, probability)))
    st.caption(
        f"Estimated churn probability: {probability:.2%} · "
        f"Estimated stay probability: {1 - probability:.2%}"
    )

    if prediction == 1:
        st.error(
            "**Potential churn detected**\n\n"
            "Consider reviewing this customer's needs and "
            "offering appropriate retention support."
        )
    else:
        st.success(
            "**Customer predicted to stay**\n\n"
            "Maintain good service quality and monitor "
            "customer satisfaction."
        )

    st.markdown("### Customer Snapshot")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Tenure", f'{customer["tenure"]} months')
    c2.metric("Contract", customer["Contract"])
    c3.metric("Monthly Charges", f'${customer["MonthlyCharges"]:.2f}')
    c4.metric("Internet", customer["InternetService"])

    with st.expander("View complete customer input"):
        st.json(customer)

    st.caption(
        "This is a model estimate, not a guarantee of future behavior. "
        "Risk bands are illustrative and are not independently calibrated."
    )


def customer_form():
    st.markdown("### 👤 Customer Profile")
    st.caption("Enter the customer's details to generate a prediction.")

    with st.form("churn_prediction_form"):
        st.markdown("#### Personal Information")
        a, b, c, d = st.columns(4)

        with a:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with b:
            senior = st.selectbox("Senior Citizen", [0, 1])
        with c:
            partner = st.selectbox("Has Partner", ["Yes", "No"])
        with d:
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])

        st.divider()
        st.markdown("#### Account & Services")

        a, b, c = st.columns(3)

        with a:
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=72, value=12
            )
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"]
            )

        with b:
            backup = st.selectbox(
                "Online Backup", ["No", "Yes", "No internet service"]
            )
            device = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"]
            )
            support = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"]
            )
            tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            movies = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"]
            )

        with c:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
            monthly = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=50.0,
                step=1.0
            )
            total = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=1000000.0,
                value=600.0,
                step=10.0
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            type="primary",
            use_container_width=True
        )

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        # Basic consistency checks
        if internet == "No":
            internet_services = [
                security, backup, device, support, tv, movies
            ]
            if any(v != "No internet service" for v in internet_services):
                st.warning(
                    "For Internet Service = No, set online and streaming "
                    "services to 'No internet service'."
                )
                return

        if phone == "No" and multiple != "No phone service":
            st.warning(
                "When Phone Service = No, Multiple Lines should be "
                "'No phone service'."
            )
            return

        if total > 0 and tenure == 0:
            st.warning(
                "For zero tenure, please review the Total Charges value."
            )
            return

        if total > 0 and tenure > 0 and total < monthly:
            st.warning(
                "Total Charges is lower than one month of charges. "
                "Please verify the values."
            )
            return

        try:
            with st.spinner("Analyzing customer profile..."):
                response = requests.post(
                    API_URL, json=customer, timeout=20
                )

            if response.ok:
                result = response.json()
                st.session_state["last_result"] = result
                st.session_state["last_customer"] = customer
                st.success("Prediction completed successfully.")
            else:
                st.error(f"API error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. Start the backend in another "
                "terminal using: python -m uvicorn app.main:app --reload"
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")


# PAGES 
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>Customer Churn Prediction System</h1>
        <p>Turn customer information into smarter retention decisions.</p>
        <br>
        <p>Explore churn risk, analyze account details, and get a prediction
        from your trained machine learning pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Welcome to Customer Churn Prediction System")
    st.write(
        "Use this workspace to estimate whether a customer may leave "
        "the service. Start a prediction to view a detailed result."
    )

    a, b, c = st.columns(3)
    with a:
        metric_card("MODEL", "Random Forest", "Trained ML pipeline")
    with b:
        metric_card("API", "FastAPI", "Real-time prediction endpoint")
    with c:
        metric_card("INTERFACE", "Streamlit", "Interactive analytics UI")

    st.markdown("### Quick Start")
    st.info(
        "Open **Predict Churn** in the sidebar, enter customer details, "
        "and click Predict Customer Churn."
    )

    if "last_result" in st.session_state:
        st.markdown("### Latest Prediction")
        show_prediction(
            st.session_state["last_result"],
            st.session_state["last_customer"]
        )

elif page == "Predict Churn":
    st.markdown("""
    <div class="hero">
        <h1>Churn Prediction Studio</h1>
        <p>Complete the customer profile and run your ML prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    customer_form()

    if "last_result" in st.session_state:
        show_prediction(
            st.session_state["last_result"],
            st.session_state["last_customer"]
        )

elif page == "Model Information":
    st.markdown("""
    <div class="hero">
        <h1>Model Information</h1>
        <p>Understand the model and its evaluation.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model Overview")
    a, b = st.columns(2)

    with a:
        st.markdown("**Algorithm**")
        st.write("Random Forest Classifier")
        st.markdown("**Preprocessing**")
        st.write("StandardScaler + OneHotEncoder")
        st.markdown("**Feature Engineering**")
        st.write(
            "Average monthly spend, long-term contract indicator, "
            "and internet-service indicator."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with b:
        st.markdown("**Selected hyperparameters**")
        st.write("- Trees: 200")
        st.write("- Maximum depth: 10")
        st.write("- Minimum samples split: 5")
        st.write("- Class weight: balanced")
        st.markdown("**Selection objective**")
        st.write("F1 score using 5-fold cross-validation.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Test results")
    a, b, c = st.columns(3)
    with a:
        metric_card("ACCURACY", "78.13%", "Test set")
    with b:
        metric_card("ROC-AUC", "82.96%", "Test set")
    with c:
        metric_card("RECALL", "77.01%", "Test set")

    st.caption(
        "These values come from the current model evaluation. "
        "They describe the held-out test results, not a guarantee "
        "of future performance."
    )

    st.markdown("### Evaluation Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        try:
            st.image(
                "images/confusion_matrix.png",
                caption="Confusion matrix on the test set",
                use_container_width=True
            )
        except Exception:
            st.info("Run src/evaluate_model.py to generate this plot.")

    with col2:
        st.markdown("**ROC Curve**")
        try:
            st.image(
                "images/roc_curve.png",
                caption="ROC curve on the test set",
                use_container_width=True
            )
        except Exception:
            st.info("Run src/evaluate_model.py to generate this plot.")

elif page == "About":
    st.markdown("""
    <div class="hero">
        <h1>About Customer Churn Prediction System</h1>
        <p>A machine learning project for customer retention analytics.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project Architecture")
    st.code("Customer form (Streamlit)\n""        ↓ HTTP POST\n""     FastAPI \n""        ↓\n""Saved Scikit-learn pipeline\n""        ↓\n""Prediction + churn probability\n""        ↓\n""Results displayed in Streamlit",language="text"
    )

    st.markdown("### Technology Stack")
    st.write(
        "Python · Pandas · Scikit-learn · Joblib · FastAPI · "
        "Pydantic · Requests · Streamlit"
    )

    st.warning(
        "This project is a decision-support prototype. "
        "Model predictions may be wrong and should be reviewed "
        "alongside other customer information."
    )

st.divider()
st.markdown(
    "<p style='text-align:center;color:#8194b5;font-size:12px;'>"
    "Machine Learning· Customer Churn Prediction System · "
    "Built with Streamlit and FastAPI</p>",
    unsafe_allow_html=True
)

