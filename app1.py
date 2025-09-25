import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("xgb_model.pkl")

st.title("💳 Fraud Detection System")
st.write("Enter transaction details below to check if it's fraudulent.")

# Input fields (must match training features order!)
merchant = st.number_input("Merchant ID", min_value=0.0, step=1.0)
category = st.number_input("Category", min_value=0.0, step=1.0)
amt = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==1 else "Female")
city_pop = st.number_input("City Population", min_value=0.0, step=1.0)
job = st.number_input("Job (encoded)", min_value=0.0, step=1.0)
age = st.number_input("Age", min_value=0, step=1)
age_group = st.number_input("Age Group", min_value=0.0, step=1.0)
day = st.number_input("Day", min_value=1, max_value=31, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)
year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
hour = st.number_input("Hour", min_value=0, max_value=23, step=1)
minute = st.number_input("Minute", min_value=0, max_value=59, step=1)
distance_km = st.number_input("Distance (km)", min_value=0.0, step=0.01)
distance_group = st.number_input("Distance Group", min_value=0.0, step=1.0)

# Arrange features in same order as training
features = np.array([[merchant, category, amt, gender, city_pop, job, age, age_group,
                      day, month, year, hour, minute, distance_km, distance_group]])

if st.button("Predict"):
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]

    if prediction[0] == 1:
        st.error(f"🚨 Fraudulent Transaction! (Fraud Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {prob:.2f})")
