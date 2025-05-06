# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 05:27:39 2025

@author: Rajshree
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('D:/OneDrive/Desktop/Final Project/TrainedModels/diabetesTrained_model.sav', 'rb'))

# Load the saved StandardScaler
loaded_scaler = pickle.load(open('D:/OneDrive/Desktop/Final Project/TrainedModels/diabetes_scaler.sav', 'rb'))

# Function for Prediction
def diabetes_prediction(input_data):
    # Convert input_data to a NumPy array (ensure all values are floats)
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the array for a single prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data using the loaded scaler
    std_data = loaded_scaler.transform(input_data_reshaped)

    # Make the prediction
    prediction = loaded_model.predict(std_data)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Streamlit App
def main():
    # Title
    st.title('Diabetes Prediction Web App')

    # User Inputs
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # Code for Prediction
    diagnosis = ''

    # Button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            # Convert input values to float and predict
            input_values = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]
            diagnosis = diabetes_prediction(input_values)
        except ValueError:
            diagnosis = 'Please enter valid numerical values for all fields.'

    st.success(diagnosis)

# Run the Streamlit App
if __name__ == '__main__':
    main()
