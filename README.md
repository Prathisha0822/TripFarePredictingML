# TripFare 🚕 — Taxi Fare Prediction (ML + Streamlit)

TripFare is a Machine Learning project that predicts **NYC taxi trip total amount** using engineered trip features and regression models.  
The final selected model is deployed using a **Streamlit web app** for interactive fare estimation.

---

## 📌 Project Highlights

- Data Cleaning + Feature Engineering (distance, duration, time-based features)
- Exploratory Data Analysis (EDA) + Outlier handling
- Model Training and Comparison (6 regression models)
- Best model selection using: **R², MAE, MSE, RMSE**
- Model saved using **joblib**
- Streamlit UI for prediction

---

## 🎯 Final Model Features & Target

### Input Features (X)
- `pickup_day`
- `am_pm`
- `is_night`
- `trip_duration_min`
- `trip_distance`
- `passenger_count`
- `RatecodeID`
- `payment_type`

### Target (y)
- `total_amount`

---

## 📊 Model Comparison (Results Summary)

Models trained and evaluated:
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Decision Tree Regressor  
- Gradient Boosting Regressor  
- Random Forest Regressor  

Best model selected based on highest **R²** and lowest **RMSE/MAE**.

✅ **Random Forest performed best** in this project.

---

## UI Snapshot:

<img width="892" height="828" alt="image" src="https://github.com/user-attachments/assets/713bfe3f-3c68-458a-af6e-7b5a25a99e59" />


