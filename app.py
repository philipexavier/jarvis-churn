import streamlit as st
import pandas as pd
from st_connection import StConnection
import joblib  # Para modelo


@st.cache_data
def load_data():
    conn = st.connection("postgresql", type="sql")
    df_net = conn.query("SELECT * FROM netflix_churn LIMIT 10000")
    df_bank = conn.query("SELECT * FROM bank_churn LIMIT 10000")
    return pd.concat([df_net, df_bank], ignore_index=True)  # União simples


st.title("Jarvis: Churn Advisor")
df = load_data()
st.dataframe(df.head())

# Predição (futuro: XGBoost)
if st.button("Treinar & Predizer"):
    # Código ML aqui
    st.success("Churn prob: 75%. Sugestão: Upsell Premium!")
