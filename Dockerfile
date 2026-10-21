# Recommendation API image (FastAPI + TensorFlow query tower + FAISS).
# The data / model folder is mounted at runtime (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HM_DATA_PATH=/app/data/hm_project \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app
RUN pip install --upgrade pip && pip install \
        "tensorflow-cpu>=2.16" "faiss-cpu>=1.8" "fastapi>=0.110" "uvicorn[standard]>=0.29" \
        "redis>=5.0" "fakeredis>=2.23" "pandas>=2.2.3" "pyarrow>=15.0" "numpy>=1.26" "pillow>=10.0" "requests>=2.31"

COPY src ./src

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if b'\"ok\"' in urllib.request.urlopen('http://localhost:8000/health').read() else 1)"
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
