import streamlit as st
import requests

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")

st.title("🔍 Fraud Detection System")
st.markdown("---")
st.markdown("Enter the transaction details below to check if it is fraudulent.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Info")
    Time = st.number_input("Time:", min_value=0, format="%.2f")
    Amount = st.number_input("Amount ($):")

with col2:
    st.subheader("PCA Features")
    features = {}
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[(i - 1) % 4]:
            features[f"V{i}"] = st.number_input(f"V{i}:", format="%.4f")

st.markdown("---")

data = {"Time": Time} | features | {"Amount": Amount}

if st.button("🔎 Predict", use_container_width=True):
    response = requests.post('https://fraud-detection-api-ayby.onrender.com/Credit_features', json=data)
    result = response.json()

    if result["prediction"] == 0:
        st.success("✅ No fraud detected — This transaction appears legitimate.")
    else:
        st.error("🚨 Fraud detected! — This transaction is suspicious.")