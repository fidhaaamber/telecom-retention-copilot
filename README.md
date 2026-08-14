# Telecom Customer Churn Prediction & Retention Copilot

An analyst-assist application for Cell2Cell churn prediction. It scores risk, shows customer-level directional drivers, prioritises a work queue, and generates grounded retention recommendations. It never contacts a customer or approves a discount.

## Run

```powershell
# Use Python 3.10-3.13. Do not use Python 3.14: ydata-profiling does not support it.
# On this computer, Python 3.12 is installed at the path below.
& 'C:\Users\FIDHAA AMBER S\AppData\Local\Programs\Python\Python312\python.exe' -m venv .capstone-py312
.\.capstone-py312\Scripts\python -m pip install --upgrade pip
.\.capstone-py312\Scripts\python -m pip install -r requirements.txt
Expand-Archive -LiteralPath 'data\archive (1).zip' -DestinationPath 'data\raw' -Force
.\.capstone-py312\Scripts\python -m scripts.train
.\.capstone-py312\Scripts\python -m scripts.tune_lightgbm
.\.capstone-py312\Scripts\python -m scripts.profile_data
.\.capstone-py312\Scripts\python -m scripts.shap_report
.\.capstone-py312\Scripts\uvicorn api.main:app --reload
# In another terminal
.\.capstone-py312\Scripts\streamlit run app\streamlit_app.py
```

Visit `http://localhost:8000/docs` for FastAPI documentation. The Streamlit URL is printed in the second terminal.

## Delivered requirements

- Reproducible SHA-256 data check and data-quality report.
- Automated ydata profile (when installed), data dictionary, EDA findings and ten reproducible business charts.
- Leakage-safe feature engineering, stratified validation, Logistic Regression/Random Forest/LightGBM comparison, champion bundle, risk threshold and scored-customer queue.
- API endpoints: `/health`, `/predict`, `/predict/batch`, `/explain`, and `/recommend`.
- Four Streamlit screens: Portfolio/EDA, Single Customer Risk, High-Risk Queue, Retention Copilot.
- Pydantic-validated grounded agent output and safe rule fallback when Ollama is unavailable.

## Local Llama 3 demo

Install Ollama, run `ollama pull llama3.1:8b`, then set `$env:USE_OLLAMA='true'` before starting the API. The copilot uses only submitted customer facts, model drivers and approved rules. It falls back clearly if Ollama is unavailable.

## Verification

```powershell
.\.capstone-py312\Scripts\pytest -q
```

## Cloud portfolio deployment

This project deploys as two services: Streamlit Community Cloud hosts the UI, while a Docker host such as Render hosts FastAPI and the trained model. The production API uses LightGBM's native contribution output, keeping the Docker image compact. Local Ollama remains a local-demo feature; the deployed API uses the approved-rules fallback unless an Ollama service is configured.

1. Create a GitHub repository and push this project, including `models/champion.joblib`, `reports/scored_customers.csv`, and `data/raw/cell2celltrain.csv`.
2. In Render, create a Blueprint from the repository. It detects `render.yaml`; once deployed, copy its HTTPS URL.
3. In Streamlit Community Cloud, deploy from the same GitHub repository with entrypoint `app/streamlit_app.py`. UI dependencies are in `app/requirements.txt`.
4. In Streamlit **App settings → Secrets**, paste:

   ```toml
   API_URL = "https://your-api-service.onrender.com"
   ```

5. Redeploy the UI and validate `/health`, prediction, queue export, and the Copilot fallback.
