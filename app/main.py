from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "tuned_random_model.pkl"
model = joblib.load(MODEL_PATH)

class customerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


app = FastAPI(
    title="Customer Churn Prediction API",
    description = "API to predict wheather a customer is likely to churn",
    version="1.0.0"
)

@app.get("/")
def home():
    return{
        "message": "Customer Churn Prediction API is running",
        "status":"success"
    }

@app.get("/health")
def health_check():
    return{
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_churn(customer: customerData):
    customer_df = pd.DataFrame([customer.model_dump()])

    prediction = int(model.predict(customer_df)[0])
    churn_probability = float(model.predict_proba(customer_df)[0][1])
    return {
        "prediction": prediction,
        "result": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(churn_probability, 4),
        "recommendation": (
            "Consider customer retention actions."
            if prediction == 1
            else "Continue regular customer engagement."
        )
    }
