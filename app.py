import streamlit as st
import joblib
import numpy as np
from sklearn.utils.validation import check_is_fitted

# --------------------------------------------------
# Page Configuration (MUST be first)
# --------------------------------------------------
st.set_page_config(
    page_title="TripFare • Urban Taxi Fare Prediction",
    layout="centered"
)

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------


@st.cache_resource
def load_model():
    return joblib.load("gradient_boosting_fare_model.pkl")


model = load_model()

# --------------------------------------------------
# Global Styling
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0f1117;
}

h1 {
    text-align: center;
    font-size: 3rem;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    color: #9aa4bf;
    margin-bottom: 2.5rem;
    font-size: 1.05rem;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    margin-top: 1.8rem;
    margin-bottom: 0.6rem;
}

.fare-container {
    margin-top: 2.5rem;
    padding: 2rem;
    border-radius: 16px;
    background: linear-gradient(90deg, #1f2937, #111827);
    text-align: center;
}

.fare-value {
    font-size: 3.4rem;
    font-weight: 700;
    color: #4ade80;
}

.fare-caption {
    color: #9aa4bf;
    font-size: 0.9rem;
    margin-top: 0.4rem;
}

.stButton>button {
    background: linear-gradient(90deg, #ffcc70, #ff6f61);
    color: black;
    font-weight: 700;
    border-radius: 12px;
    height: 3.1rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
st.markdown("""
<h1>🚕 TripFare</h1>
<p class="subtitle">
Urban taxi fare prediction powered by Machine Learning and real-world trip data
</p>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

# Passenger Count (UI-only)
st.markdown('<div class="section-title">👥 Passenger Count</div>',
            unsafe_allow_html=True)
passenger_count = st.selectbox(
    "Passenger Count",
    options=[1, 2, 3, 4, 5, 6],
    index=0,
    label_visibility="collapsed"
)

# Trip Distance
st.markdown('<div class="section-title">📍 Trip Distance (km)</div>',
            unsafe_allow_html=True)
trip_distance = st.slider(
    "Trip Distance",
    min_value=0.5,
    max_value=50.0,
    value=5.0,
    step=0.5,
    label_visibility="collapsed"
)

# Trip Duration
st.markdown('<div class="section-title">⏱️ Trip Duration (minutes)</div>',
            unsafe_allow_html=True)
trip_duration = st.slider(
    "Trip Duration",
    min_value=2,
    max_value=120,
    value=15,
    step=1,
    label_visibility="collapsed"
)

# Pickup Time (AM / PM based)
st.markdown('<div class="section-title">🕒 Pickup Time</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    pickup_hour_12 = st.selectbox(
        "Hour",
        options=list(range(1, 13)),
        index=11
    )

with col2:
    pickup_minute = st.slider(
        "Minute",
        min_value=0,
        max_value=59,
        value=0,
        step=1
    )

with col3:
    am_pm = st.selectbox(
        "AM / PM",
        options=["AM", "PM"],
        index=1
    )

# --------------------------------------------------
# Time Conversion Logic (CRITICAL)
# --------------------------------------------------

# Convert 12-hour format to 24-hour format
if am_pm == "AM":
    pickup_hour_24 = 0 if pickup_hour_12 == 12 else pickup_hour_12
else:
    pickup_hour_24 = 12 if pickup_hour_12 == 12 else pickup_hour_12 + 12

# Convert to fractional hour for ML model
pickup_time = pickup_hour_24 + (pickup_minute / 60)

# Automatic Night / Early Morning Detection
is_night = (
    (pickup_hour_24 >= 22) or   # 10 PM - 11:59 PM
    (pickup_hour_24 <= 5)       # 12 AM - 5:59 AM
)

# Display Night Status (read-only)
night_label = "🌙 Night / Early Morning Ride" if is_night else "☀️ Daytime Ride"
st.caption(f"Detected Ride Time: **{night_label}**")

# --------------------------------------------------
# ACTION
# --------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button(
    "✨ Preview Estimated Fare", use_container_width=True)

# --------------------------------------------------
# PREDICTION OUTPUT
# --------------------------------------------------
if predict_clicked:
    check_is_fitted(model)

    # Model expects ONLY trained features
    input_data = np.array(
        [[trip_distance, trip_duration, pickup_time,
            int(is_night), passenger_count]],
        dtype=float
    )

    with st.spinner("Analyzing historical trip patterns..."):
        predicted_fare = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="fare-container">
        <div class="fare-value">${predicted_fare:.2f}</div>
        <div class="fare-caption">
            Estimated for a {trip_distance} km trip taking approximately {trip_duration} minutes<br>
            with {passenger_count} passenger{'s' if passenger_count > 1 else ''} at
            {pickup_hour_12:02d}:{pickup_minute:02d} {am_pm}
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<p style="text-align:center; color:#6b7280; font-size:0.8rem; margin-top:3rem;">
Built using Exploratory Data Analysis, Feature Engineering, Regression Modeling,<br>
and Hyperparameter Tuning • Streamlit Deployment<br><br>
Urban Transportation Analytics & Predictive Modeling Project
</p>
""", unsafe_allow_html=True)
