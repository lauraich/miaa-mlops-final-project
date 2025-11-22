FROM python:3.11-slim

WORKDIR /app

COPY app/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

EXPOSE 8080

ENV ENVIRONMENT=dev
# En producción/CI establecer MODEL_BUCKET, MODEL_KEY, AWS credentials como secrets/env

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
