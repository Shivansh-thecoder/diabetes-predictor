from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = os.path.join(os.path.dirname(__file__),'..' ,'models', 'dt_model.pkl')
model = pickle.load(open(MODEL_PATH, 'rb'))

app = FastAPI(title="Diabetes Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during dev; restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Patient(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def root():
    return {"message": "Diabetes Prediction API. Use POST /predict with patient fields."}

@app.post("/predict")
def predict(patient: Patient):
    data = pd.DataFrame([patient.dict()])
    prediction = int(model.predict(data)[0])
    proba = float(model.predict_proba(data)[0][prediction])
    return {"prediction": prediction, "probability": proba}
