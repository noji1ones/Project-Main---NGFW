FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# This is for th model and mount it via volume
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
