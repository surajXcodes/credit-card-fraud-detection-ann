import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

# Load trained model
model = tf.keras.models.load_model("models/ann_model.h5")

st.set_page_config(page_title="Fraud Detection App", page_icon="💳")

st.title("💳 Credit Card Fraud Detection App")

st.write(
"""
This app uses a trained Artificial Neural Network (ANN) model  
to detect whether a credit card transaction is **Fraudulent** or **Genuine**.
"""
)

# Sample dataset loader
@st.cache_data
def load_sample():
    df = pd.read_csv("data/creditcard.csv")
    return df.sample(1)

# Button to autofill sample
if st.button("Load Sample Transaction"):
    sample = load_sample()
    st.session_state.sample_values = sample.drop("Class", axis=1).values.flatten()

inputs = []

st.subheader("Enter Transaction Features")

for i in range(30):

    default = 0.0

    if "sample_values" in st.session_state:
        default = float(st.session_state.sample_values[i])

    val = st.number_input(f"Feature {i+1}", value=default)

    inputs.append(val)

# Prediction
if st.button("Predict Transaction"):

    data = np.array(inputs).reshape(1,-1)

    probability = model.predict(data)[0][0]

    fraud_prob = float(probability)

    st.subheader("Prediction Result")

    st.write(f"Fraud Probability: **{fraud_prob:.2%}**")

    st.progress(int(fraud_prob * 100))

    if fraud_prob > 0.8:
        st.error("🚨 Fraud Transaction Detected")
    else:
        st.success("✅ Genuine Transaction")