FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install streamlit pandas sqlalchemy psycopg2-binary scikit-learn xgboost st-connection
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
