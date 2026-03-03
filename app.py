import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib


@st.cache_data
def load_data():
    import psycopg2
    from sqlalchemy import create_engine

    engine = create_engine(st.secrets["postgres"])
    df_net = pd.read_sql(
        "SELECT age, subscriptiontype, watchhours, lastlogindays, churn FROM netflix_churn LIMIT 5000",
        engine,
    )
    df_bank = pd.read_sql(
        "SELECT Age, NumOfProducts, Balance, Exited as churn FROM bank_churn LIMIT 5000",
        engine,
    )
    df_net.columns = ["age", "plan", "watchhours", "days_inactive", "churn"]
    df_bank.columns = ["age", "num_products", "balance", "churn"]
    df = pd.concat(
        [
            df_net,
            df_bank.assign(
                plan="Standard", watchhours=df_bank["balance"] / 10000, days_inactive=30
            ),
        ],
        ignore_index=True,
    )
    return df


st.title("🦾 Jarvis: Churn Advisor")
df = load_data()
st.metric("Total Clientes", len(df))
st.metric("Churn Rate", f"{df['churn'].mean():.1%}")

if st.button("Treinar Modelo"):
    features = ["age", "watchhours", "days_inactive"]
    X = df[features].fillna(0)
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = XGBClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    st.success(f"Modelo treinado! Accuracy: {acc:.1%}")
    joblib.dump(model, "model.pkl")

# Predição Interativa
st.subheader("Predição Personalizada")
age = st.slider("Idade", 18, 80, 40)
watch = st.slider("Horas Uso/Mês", 0.0, 50.0, 10.0)
days = st.slider("Dias Inativo", 0, 60, 20)
if st.button("Analisar"):
    model = joblib.load("model.pkl")
    pred = model.predict([[age, watch, days]])[0]
    prob = model.predict_proba([[age, watch, days]])[:, 1][0]
    st.metric("Churn Prob", f"{prob:.1%}")
    if prob > 0.7:
        st.error("🚨 ALTO RISCO: Envie oferta upsell Premium!")
    else:
        st.success("✅ Baixo risco: Manter engajamento.")
