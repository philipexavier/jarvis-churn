import streamlit as st
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# =========================
# Config DB (Supabase / Postgres)
# =========================

DB_PROJ = os.getenv("DB_PROJ")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

DB_URL = f"postgresql://postgres.{DB_PROJ}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


@st.cache_data
def load_data():
    engine = create_engine(DB_URL)

    # Netflix
    df_net = pd.read_sql(
        """
        SELECT
            age,
            subscription_type AS plan,
            watch_hours,
            last_login_days AS days_inactive,
            churned AS churn
        FROM netflix_churn
        LIMIT 5000
        """,
        engine,
    )

    # Banco
    df_bank = pd.read_sql(
        """
        SELECT
            Age AS age,
            NumOfProducts AS num_products,
            Balance / 10000 AS watch_hours,
            30 AS days_inactive,
            Exited AS churn
        FROM bank_churn
        LIMIT 5000
        """,
        engine,
    )

    # Normalizar tipos numéricos (tratando strings vazias e valores sujos)
    numeric_cols = ["age", "watchhours", "days_inactive", "churn"]
    for col in numeric_cols:
        for df_ in (df_net, df_bank):
            if col in df_.columns:
                df_[col] = pd.to_numeric(df_[col], errors="coerce")

    # Garantir churn como 0/1 int (NaN -> 0)
    df_net["churn"] = df_net["churn"].fillna(0).astype(int)
    df_bank["churn"] = df_bank["churn"].fillna(0).astype(int)

    # Concatenar e preencher NaN restantes com 0
    df = pd.concat([df_net, df_bank], ignore_index=True).fillna(0)

    return df


# =========================
# UI Streamlit
# =========================

st.title("🦾 Jarvis: Churn Advisor")

df = load_data()

col1, col2 = st.columns(2)
col1.metric("Total Clientes", len(df))
col2.metric("Churn Rate", f"{df['churn'].mean():.1%}")

# =========================
# Treino do modelo
# =========================

if st.button("Treinar XGBoost"):
    features = ["age", "watch_hours", "days_inactive"]
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

# =========================
# Predição
# =========================

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

st.subheader("Amostra de Dados")
st.dataframe(df.head())
