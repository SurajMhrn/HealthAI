# App image: API + UI. CPU only. RAGAS and its LangChain dependencies are not included.
# Data, indexes and .env are mounted at run time (see docker-compose.yml); nothing secret is baked in.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY healthrag/ healthrag/
COPY ui/ ui/
COPY scripts/ scripts/
COPY config.yaml .

RUN useradd --create-home appuser && mkdir -p /app/.cache /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "healthrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
