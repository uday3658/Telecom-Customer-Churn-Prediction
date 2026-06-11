# ── Stage: runtime ──────────────────────────────────────────
FROM python:3.10-slim

# # Prevent .pyc files and buffer stdout/stderr
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

WORKDIR /app



# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# (Optional) copy pre-trained model if it exists
# COPY model.pkl .

# Data is mounted at runtime — see docker run / docker-compose below
# COPY Telco-Customer-Churn.csv .

EXPOSE 8000

# Run with uvicorn — 1 worker is enough for inference; scale up if needed

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
