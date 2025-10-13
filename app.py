# Importing required libraries/packages
import streamlit as st
import pandas as pd
import joblib

# Loading the trained model
model = joblib.load("forest_best_model.pkl")

# Adding an app title
st.title("Cardiovascular Disease Prediction")

# Inputting fields
age = st.number_input("Age (days)", min_value=0)
weight = st.number_input("Weight (kg)", min_value=0.0)
height = st.number_input("Height (cm)", min_value=0.0)
ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=0)
ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=0)
cholesterol = st.selectbox("Cholesterol Level (1: normal, 2: above normal, 3: well above normal)", [1, 2, 3])


# Creating a predict button
if st.button("Predict"):
    # Calculate BMI
    if height > 0:
        bmi = round(weight / ((height / 100) ** 2), 2)
    else:
        st.error("Height must be greater than 0 to calculate BMI.")
        st.stop()

    # Developing input DataFrame
    input_df = pd.DataFrame([{
        "age": age,
        "weight": weight,
        "height": height,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "bmi": bmi}])

    # Creating prediction
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {'Cardiovascular Disease' if prediction == 1 else 'No Disease'}")

