# HF Spaces Docker SDK expects the app to listen on port 7860
FROM python:3.10-slim

WORKDIR /app

# system deps for opencv/ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py app.py start.sh ./
COPY best.pt ./best.pt

RUN chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]
