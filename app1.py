import streamlit as st
import pickle
import numpy as np

# Load the scaler and model
scaler = pickle.load(open("./model/scaler.pkl", "rb"))
model = pickle.load(open("./model/Logistic.pkl", "rb"))

# Function to predict diabetes
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    prediction = model.predict(new_data)
    return 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

# Streamlit app
st.title("Diabetes Prediction")

st.sidebar.header("Input Parameters")
Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
Glucose = st.sidebar.number_input("Glucose", min_value=0.0, max_value=200.0, value=120.0)
BloodPressure = st.sidebar.number_input("BloodPressure", min_value=0.0, max_value=150.0, value=70.0)
SkinThickness = st.sidebar.number_input("SkinThickness", min_value=0.0, max_value=100.0, value=20.0)
Insulin = st.sidebar.number_input("Insulin", min_value=0.0, max_value=1000.0, value=79.0)
BMI = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=32.0)
DiabetesPedigreeFunction = st.sidebar.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5)
Age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25)

if st.sidebar.button("Predict"):
    result = predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    st.write(f"The person is: {result}")

st.sidebar.write("Adjust the sliders and click 'Predict' to get a result.")