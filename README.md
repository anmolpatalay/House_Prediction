House Price Predictor

A small end-to-end machine learning project — from raw data to a deployed, containerized API. Built to practice the full pipeline: model training, backend API design, and deployment, not just model accuracy in isolation.

Overview

Takes structured housing features (area, bedrooms, bathrooms, amenities, etc.) and returns a predicted sale price using a trained regression model, served through a FastAPI backend with a simple HTML frontend.

Tech Stack
Model: scikit-learn (Random Forest Regressor) with a preprocessing pipeline for categorical encoding
Backend: FastAPI, Pydantic for request validation
Frontend: Plain HTML/CSS/JS, served as static files by FastAPI
Containerization: Docker
Data: Housing Prices Dataset (Kaggle)
Features
Trained ML pipeline that bundles preprocessing and model into a single serialized artifact
REST API with input validation via Pydantic schemas
Interactive API documentation via FastAPI's built-in Swagger UI (/docs)
Simple form-based frontend that calls the API directly
Fully containerized — runs identically locally and in deployment
CORS-enabled for frontend/backend communication across origins
Project Structure
house-predictor/
├── backend/
│   └── main.py          # FastAPI app, prediction endpoint, static file serving
├── model/
│   ├── train.py          # Data preprocessing + model training script
│   └── house_model.joblib
├── frontend/
│   └── index.html         # Form UI that calls the /predict endpoint
├── Dockerfile
├── requirements.txt
└── README.md
Running Locally

Without Docker:

bash
pip install -r requirements.txt
cd model && python train.py      # trains and saves the model
cd ../backend && uvicorn main:app --reload

Visit http://localhost:8000/static/index.html

With Docker:

bash
docker build -t house-predictor .
docker run -p 8000:8000 house-predictor

Visit http://localhost:8000/static/index.html

API

POST /predict

Request body:

json
{
  "area": 7420,
  "bedrooms": 4,
  "bathrooms": 2,
  "stories": 3,
  "mainroad": "yes",
  "guestroom": "no",
  "basement": "no",
  "hotwaterheating": "no",
  "airconditioning": "yes",
  "parking": 2,
  "prefarea": "yes",
  "furnishingstatus": "furnished"
}

Response:

json
{
  "predicted_price": 12250000.0
}
What This Project Demonstrates
Building a reusable ML pipeline (preprocessing + model) rather than a standalone script
Designing a validated REST API around a trained model
Structuring a project for containerized deployment (correct path handling, dependency isolation, layer caching)
Debugging real deployment issues: build context size, path resolution inside containers, dependency version conflicts across Python versions
