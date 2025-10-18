# 🩺 Diabetes Predictor — FastAPI (Backend) + Streamlit (Frontend)

A small full-stack ML demo that predicts diabetes risk using a Decision Tree model.
- **Backend:** FastAPI (`backend/main.py`) — serves a `/predict` endpoint.
- **Frontend:** Streamlit (`frontend/application.py`) — user-facing UI that calls the API.
- **Model:** `models/dt_model.pkl` (trained on the PIMA Indians Diabetes dataset).

---

## 🚀 Features
- Simple, interpretable Decision Tree classifier
- Clean separation of frontend (UI) and backend (model API)
- Easily deployable (Streamlit Cloud for frontend, Render/Heroku for backend)
- Example input validation via Pydantic

---