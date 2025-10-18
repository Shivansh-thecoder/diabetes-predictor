import streamlit as st
import requests

st.set_page_config(page_title="Diabetes Predictor",layout='centered')
st.title("Diabetes Prediction (Decision Tree via FastAPI)")

st.write('Enter patient details:')

def user_input():
    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    glucose = st.number_input("Glucose", 0, 200, 120)
    bp = st.number_input("Blood Pressure", 0, 180, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 1, 120, 30)
    return {
        "Pregnancies": int(pregnancies),
        "Glucose": float(glucose),
        "BloodPressure": float(bp),
        "SkinThickness": float(skin),
        "Insulin": float(insulin),
        "BMI": float(bmi),
        "DiabetesPedigreeFunction": float(dpf),
        "Age": int(age)
    }

input_data=user_input()

API_URL=st.text_input("API URL", value="http://127.0.0.1:8000/predict")##FastAPI endpoint

if st.button('PREDICT'):
    try:
           with st.spinner("Calling API..."):
            res = requests.post(API_URL, json=input_data, timeout=10)
            res.raise_for_status()
            r = res.json()
            if r.get("prediction") == 1:
                st.error(f"Prediction: Patient at RISK of diabetes.\nProbability: {r.get('probability')*100:.2f}%")
            else:
                st.success(f"Prediction: Patient is NOT at risk.\nProbability: {r.get('probability')*100:.2f}%")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")