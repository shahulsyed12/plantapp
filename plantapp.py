import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load the trained model
with open('random_forest_model_plant2.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a function to make predictions
def predict_ac_power(input_data):
    prediction = model.predict(input_data)
    return prediction

# Define the Streamlit app layout and functionality
def main():
    st.set_page_config(
        page_title="Solar Power Prediction",
        page_icon=":sunny:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Solar Power Generation Prediction")
    st.markdown("""
    <style>
        .main {background-color: #f0f0f0;}
        .stButton>button {background-color: #4CAF50; color:white;}
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Input Parameters")
    st.sidebar.markdown("Provide the input parameters to get the AC Power prediction.")

    # Input fields for the features
    dc_power = st.sidebar.number_input("DC Power", min_value=0.0, format="%.2f")
    irradiation = st.sidebar.number_input("Irradiation", min_value=0.0, format="%.2f")
    ambient_temperature = st.sidebar.number_input("Ambient Temperature (°C)", min_value=-40.0, max_value=60.0, format="%.2f")
    module_temperature = st.sidebar.number_input("Module Temperature (°C)", min_value=-40.0, max_value=80.0, format="%.2f")
    hour = st.sidebar.number_input("Hour", min_value=0, max_value=23, format="%d")
    day = st.sidebar.number_input("Day", min_value=1, max_value=31, format="%d")
    month = st.sidebar.number_input("Month", min_value=1, max_value=12, format="%d")
    year = st.sidebar.number_input("Year", min_value=2000, max_value=datetime.now().year, format="%d")

    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'DC_POWER': [dc_power],
        'IRRADIATION': [irradiation],
        'AMBIENT_TEMPERATURE': [ambient_temperature],
        'MODULE_TEMPERATURE': [module_temperature],
        'hour': [hour],
        'day': [day],
        'month': [month],
        'year': [year]
    })

    # Prediction button
    if st.sidebar.button("Predict AC Power"):
        prediction = predict_ac_power(input_data)
        st.success(f"Predicted AC Power: {prediction[0]:.2f} kW")

if __name__ == "__main__":
    main()
