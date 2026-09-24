FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PRISM_DEVICE=cpu PRISM_ENABLE_RERANKER=0
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY api.py .
COPY src/ src/
COPY web/ web/
COPY runtime_index/ runtime_index/
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
