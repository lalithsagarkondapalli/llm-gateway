FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
RUN pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" "httpx>=0.27" "pydantic>=2.6" \
    "pyyaml>=6.0" "numpy>=1.26" "prometheus-client>=0.20"

COPY gateway ./gateway
COPY config*.yaml ./

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "gateway"]
