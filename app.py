import streamlit as st
import joblib
import numpy as np
from sklearn.utils.validation import check_is_fitted

st.set_page_config(
    page_title="TripFare • Urban Taxi Fare Prediction",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("taxi_fare_prediction_model.joblib")

model = load_model()

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

st.markdown("""
<h1>🚕 TripFare</h1>
<p class="subtitle">
Urban taxi fare prediction powered by Machine Learning and real-world trip data
</p>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">👥 Passenger Count</div>', unsafe_allow_html=True)
passenger_count = st.selectbox(
    "Passenger Count",
    options=[1, 2, 3, 4, 5, 6],
    index=0,
    label_visibility="collapsed"
)

st.markdown('<div class="section-title">📍 Trip Distance (km)</div>', unsafe_allow_html=True)
trip_distance = st.slider(
    "Trip Distance",
    min_value=0.5,
    max_value=50.0,
    value=5.0,
    step=0.5,
    label_visibility="collapsed"
)

st.markdown('<div class="section-title">⏱️ Trip Duration (minutes)</div>', unsafe_allow_html=True)
trip_duration = st.slider(
    "Trip Duration",
    min_value=2,
    max_value=120,
    value=15,
    step=1,
    label_visibility="collapsed"
)

st.markdown('<div class="section-title">📅 Pickup Day</div>', unsafe_allow_html=True)
pickup_day = st.selectbox(
    "Pickup Day",
    options=["weekday", "weekend"],
    index=0,
    label_visibility="collapsed"
)
pickup_day_enc = 0 if pickup_day == "weekday" else 1

st.markdown('<div class="section-title">🕒 Pickup Time</div>', unsafe_allow_html=True)
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

if am_pm == "AM":
    pickup_hour_24 = 0 if pickup_hour_12 == 12 else pickup_hour_12
else:
    pickup_hour_24 = 12 if pickup_hour_12 == 12 else pickup_hour_12 + 12

is_night = ((pickup_hour_24 >= 22) or (pickup_hour_24 <= 5))
night_label = "🌙 Night / Early Morning Ride" if is_night else "☀️ Daytime Ride"
st.caption(f"Detected Ride Time: **{night_label}**")

am_pm_enc = 0 if am_pm == "AM" else 1
is_night_enc = int(is_night)

st.markdown('<div class="section-title">🏷️ Rate Code</div>', unsafe_allow_html=True)
ratecode_id = st.selectbox(
    "Rate Code",
    options=[1, 2, 3, 4, 5, 6],
    index=0,
    label_visibility="collapsed"
)

st.markdown('<div class="section-title">💳 Payment Type</div>', unsafe_allow_html=True)
payment_type = st.selectbox(
    "Payment Type",
    options=[1, 2, 3, 4],
    index=0,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("✨ Preview Estimated Fare", use_container_width=True)

if predict_clicked:
    check_is_fitted(model)

    input_data = np.array(
        [[
            pickup_day_enc,
            am_pm_enc,
            is_night_enc,
            float(trip_duration),
            float(trip_distance),
            float(passenger_count),
            float(ratecode_id),
            float(payment_type)
        ]],
        dtype=float
    )

    with st.spinner("Analyzing historical trip patterns..."):
        predicted_fare = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="fare-container">
        <div class="fare-value">${predicted_fare:.2f}</div>
        <div class="fare-caption">
            This is the estimated fare for the trip you are planning to book.<br>
            A {trip_distance} km ride taking about {trip_duration} minutes,<br>
            with {passenger_count} passenger{'s' if passenger_count > 1 else ''}, scheduled on a {pickup_day}<br>
            at {pickup_hour_12:02d}:{pickup_minute:02d} {am_pm}.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<p style="text-align:center; color:#6b7280; font-size:0.8rem; margin-top:3rem;">
<br>Urban Transportation Analytics & Predictive Modeling Project
</p>
""", unsafe_allow_html=True)
