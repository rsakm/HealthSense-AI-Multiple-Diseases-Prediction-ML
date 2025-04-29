# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 08:12:43 2025

@author: Rajshree
"""
import numpy as np
import pickle
import streamlit as st

# Load the trained model and scaler
model = pickle.load(open("D:/OneDrive/Desktop/Final Project/TrainedModels/parkinsons_model.sav", "rb"))
scaler = pickle.load(open("D:/OneDrive/Desktop/Final Project/TrainedModels/parkinsons_scaler.sav", "rb"))

# Function for Prediction
def parkinsons_prediction(input_data):
    input_data_np = np.asarray(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data_np)
    prediction = model.predict(input_data_scaled)
    
    return "🟥 The person has Parkinson's Disease" if prediction[0] == 1 else "🟩 The person does NOT have Parkinson's Disease"

# Streamlit Web App
def main():
    st.set_page_config(page_title="Parkinson's Disease Prediction", layout="wide")
    st.title("🧠 Parkinson's Disease Prediction Web App")
    st.write("Enter the required biomedical features to predict if the person has Parkinson’s Disease.")

    # Feature Names
    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", 
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", 
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", 
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", 
        "spread1", "spread2", "D2", "PPE"
    ]

    # Create input fields in two columns
    input_values = []
    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            value = st.text_input(f"{feature}")
            input_values.append(value)

    diagnosis = ''

    if st.button("🔍 Predict"):
        try:
            # Convert input values to float
            input_data = list(map(float, input_values))
            diagnosis = parkinsons_prediction(input_data)
        except ValueError:
            diagnosis = "⚠️ Please enter valid numerical values for all fields."

    st.success(diagnosis)

if __name__ == '__main__':
    main()

