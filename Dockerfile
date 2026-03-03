FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --upgrade pip
RUN pip install streamlit==1.38.0 pandas==2.2.2 sqlalchemy==2.0.35 psycopg2-binary==2.9.9 scikit-learn==1.5.1 xgboost==2.1.1
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
