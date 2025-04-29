# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 07:12:06 2025

@author: Rajshree
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained model and scaler
model = pickle.load(open("D:/OneDrive/Desktop/Final Project/TrainedModels/heart_disease_model.sav", 'rb'))
scaler = pickle.load(open('D:/OneDrive/Desktop/Final Project/TrainedModels/heart_scaler.sav', 'rb'))

# Function for prediction
def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data
    input_data_scaled = scaler.transform(input_data_reshaped)

    # Prediction
    prediction = model.predict(input_data_scaled)

    return "The person does not have Heart Disease" if prediction[0] == 0 else "The person has Heart Disease"

# Streamlit UI
def main():
    st.title("Heart Disease Prediction Web App")

    # Input fields
    age = st.text_input("Age")
    sex = st.text_input("Sex (1 = Male, 0 = Female)")
    cp = st.text_input("Chest Pain Type (0-3)")
    trestbps = st.text_input("Resting Blood Pressure")
    chol = st.text_input("Cholesterol Level")
    fbs = st.text_input("Fasting Blood Sugar (1 = True, 0 = False)")
    restecg = st.text_input("Resting ECG Results (0-2)")
    thalach = st.text_input("Max Heart Rate Achieved")
    exang = st.text_input("Exercise-Induced Angina (1 = Yes, 0 = No)")
    oldpeak = st.text_input("ST Depression Induced")
    slope = st.text_input("Slope of Peak Exercise ST (0-2)")
    ca = st.text_input("Number of Major Vessels (0-3)")
    thal = st.text_input("Thalassemia Type (0-3)")

    diagnosis = ""

    if st.button("Predict"):
        try:
            # Convert input data to float
            input_data = [float(age), float(sex), float(cp), float(trestbps), float(chol),
                          float(fbs), float(restecg), float(thalach), float(exang),
                          float(oldpeak), float(slope), float(ca), float(thal)]
            
            diagnosis = heart_disease_prediction(input_data)
        except:
            diagnosis = "Please enter valid numerical values!"

    st.success(diagnosis)

if __name__ == '__main__':
    main()
