FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --upgrade pip && \
    pip install streamlit==1.38.0 pandas==2.2.2 sqlalchemy==2.0.35 psycopg2-binary==2.9.9 scikit-learn==1.5.1 xgboost==2.1.1 joblib==1.4.2 numpy==1.26.4
# Garante .streamlit
RUN mkdir -p ~/.streamlit && touch ~/.streamlit/credentials.toml
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless=true"]
