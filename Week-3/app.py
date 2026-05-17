from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Credit Scoring API", description="API to predict loan risk level", version="1.0")

from fastapi.responses import FileResponse
import os

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model pipeline
model = joblib.load('credit_model.pkl')

# Define request schema
class LoanApplicant(BaseModel):
    Age: int
    Sex: str
    Job: int
    Housing: str
    Saving_accounts: str
    Checking_account: str
    Credit_amount: int
    Duration: int
    Purpose: str

@app.get("/")
def home():
    return {"status": "CreditGuard API Running"}


@app.post("/predict")
def predict(applicant: LoanApplicant):
    # Convert input to DataFrame (matching the pipeline's expected format)
    # Note: Column names must match exactly what was used during training
    input_df = pd.DataFrame([{
        'Age': applicant.Age,
        'Sex': applicant.Sex,
        'Job': applicant.Job,
        'Housing': applicant.Housing,
        'Saving accounts': applicant.Saving_accounts,
        'Checking account': applicant.Checking_account,
        'Credit amount': applicant.Credit_amount,
        'Duration': applicant.Duration,
        'Purpose': applicant.Purpose
    }])
    
    # Get prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].tolist()
    
    risk_level = "High Risk (Bad)" if prediction == 1 else "Low Risk (Good)"
    
    # Generate basic explainability factors
    reasons = []
    if applicant.Saving_accounts in ['little', 'moderate']:
        reasons.append(f"Low savings account")
    if applicant.Duration > 24:
        reasons.append(f"High loan duration")
    if applicant.Credit_amount > 5000:
        reasons.append(f"Large credit amount")
    if applicant.Checking_account in ['little']:
        reasons.append(f"Low checking account balance")
    if applicant.Job in [0, 1]:
        reasons.append(f"Unskilled job category")
        
    if not reasons:
        reasons.append("Overall profile characteristics")
    
    return {
        "prediction": int(prediction),
        "risk_level": risk_level,
        "probabilities": {
            "Low Risk (Good)": probability[0],
            "High Risk (Bad)": probability[1]
        },
        "reasons": reasons
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
