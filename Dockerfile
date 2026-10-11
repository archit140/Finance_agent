FROM python:3.12

WORKDIR /app

# ── Layer 1: install deps (cached until requirements.txt changes) ─────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source and local demo data ─────────────────────────────────────
COPY backend/ backend/
COPY company_kb/ company_kb/
COPY sandbox/ sandbox/
COPY documents/ documents/
COPY app.py sandbox_api.py seed_scenarios.py evaluate.py main.py .
COPY goldens.json eval_tasks.json .

ENV ERP_API_URL=http://127.0.0.1:8000 \
    ERP_DATA_DIR=/app/sandbox/data \
    PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
