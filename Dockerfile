FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /service
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY api ./api
COPY utils ./utils
COPY models/champion.joblib ./models/champion.joblib
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
