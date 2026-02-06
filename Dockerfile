FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CREDIT_RISK_MODEL=/app/artifacts/model.joblib

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[serve]"

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Mount a trained bundle at /app/artifacts, e.g. -v $(pwd)/artifacts:/app/artifacts
CMD ["uvicorn", "credit_risk.serving.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
