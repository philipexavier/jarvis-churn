import streamlit as st
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Supabase via ENV (Easypanel Secrets)
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


@st.cache_data
def load_data():
    engine = create_engine(DB_URL)
    df_net = pd.read_sql(
        "SELECT age, subscriptiontype as plan, watchhours, lastlogindays as days_inactive, churned as churn FROM netflix_churn LIMIT 5000",
        engine,
    )
    df_bank = pd.read_sql(
        "SELECT Age as age, NumOfProducts as num_products, Balance/10000 as watchhours, 30 as days_inactive, Exited as churn FROM bank_churn LIMIT 5000",
        engine,
    )
    df = pd.concat([df_net, df_bank], ignore_index=True).fillna(0)
    return df


st.title("🦾 Jarvis: Churn Advisor")
df = load_data()
col1, col2 = st.columns(2)
col1.metric("Total Clientes", len(df))
col2.metric("Churn Rate", f"{df['churn'].mean():.1%}")

if st.button("Treinar XGBoost"):
    features = ["age", "watchhours", "days_inactive"]
    X = df[features]
    y = df["churn"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    joblib.dump(model, "model.pkl")
    st.success(f"✅ Modelo treinado! Accuracy: {acc:.1%} | Salvo como model.pkl")

# Predição
st.subheader("Predição & Decisões")
age = st.slider("Idade", 18, 80, 40)
watch_h = st.slider("Horas Assistidas/Mês", 0.0, 50.0, 10.0)
days_in = st.slider("Dias sem Login", 0, 60, 20)
if st.button("Analisar Risco Churn"):
    try:
        model = joblib.load("model.pkl")
        input_data = np.array([[age, watch_h, days_in]])
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[:, 1][0]
        st.metric("Probabilidade Churn", f"{prob:.1%}")
        if prob > 0.7:
            st.error(
                "🚨 ALTO RISCO: Sugestão - Ofereça Premium grátis 1 mês + email personalizado!"
            )
        elif prob > 0.4:
            st.warning("⚠️ Risco Médio: Envie notificação push com conteúdos favoritos.")
        else:
            st.success("✅ Baixo Risco: Mantenha newsletters semanais.")
    except FileNotFoundError:
        st.info("Treine o modelo primeiro!")

st.dataframe(df.head())
