FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 & streamlit run frontend/streamlit_app.py --server.port 8501"]
