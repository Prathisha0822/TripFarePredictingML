# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Taxi Fare Predictor (INR)",
    page_icon="🚕",
    layout="centered"
)

st.title("🚕 Taxi Fare Prediction")
st.caption("Random Forest Model • Output in INR")

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = Path("fare_prediction_rf.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

USD_TO_INR = 83.0  # conversion rate

# -----------------------------
# Inputs
# -----------------------------
st.subheader("Enter Trip Details")

# Distance
distance_km = st.number_input(
    "Distance (km)",
    min_value=0.1,
    value=3.0,
    step=0.1
)

# Tip
tip_inr = st.number_input(
    "Tip Amount (INR)",
    min_value=0.0,
    value=50.0,
    step=10.0
)

# Trip Duration (time-like, vertical)
st.markdown("### ⏰ Trip Duration")

hours = st.number_input(
    "Hours",
    min_value=0,
    value=0,
    step=1,
    format="%d"
)

minutes = st.number_input(
    "Minutes",
    min_value=0,
    max_value=59,
    value=15,
    step=1,
    format="%02d"
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Fare", type="primary", use_container_width=True):

    duration_minutes = (hours * 60) + minutes

    if duration_minutes <= 0:
        st.warning("Trip duration must be greater than 0 minutes.")
        st.stop()

    # Convert INR → USD for model
    tip_usd = tip_inr / USD_TO_INR

    X_input = pd.DataFrame([{
        "duration_minutes": duration_minutes,
        "distance_km": distance_km,
        "tip_amount": tip_usd
    }])

    # Predict (model outputs USD)
    pred_usd = model.predict(X_input)[0]
    pred_inr = pred_usd * USD_TO_INR

    st.success(f"💰 **Predicted Fare Amount: ₹ {pred_inr:,.2f} INR**")

    with st.expander("Show Model Input"):
        st.write(f"Duration: **{hours:02d}:{minutes:02d} (HH:MM)**")
        st.write(f"Distance: **{distance_km} km**")
        st.write(f"Tip: **₹ {tip_inr}**")

    st.caption("Conversion used: 1 USD = 83 INR")

