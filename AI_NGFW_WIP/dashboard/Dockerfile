FROM python:3.9-slim

WORKDIR /app

# trying to copy requirements first to cache dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

#For the rest of the dashB code
COPY . .

# Run Streamlit on port 8501 - Does localhost work 
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]