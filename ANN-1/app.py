import streamlit as st
import numpy as np
import pandas as pd
import pickle  # For model loading

# Load the trained model
MODEL_PATH = "cancer.pkl"  # Updated to your pkl file
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# App title
st.title("Breast Cancer Diagnosis Predictor")

# App description
st.markdown("""
This application predicts whether the tumor is **benign (0)** or **malignant (1)** based on user inputs.
Enter the required details below and click **Predict** to see the result.
""")

# Input fields for user data
st.sidebar.header("Input Features")
radius_mean = st.sidebar.number_input("Radius Mean", value=17.99)
texture_mean = st.sidebar.number_input("Texture Mean", value=10.38)
perimeter_mean = st.sidebar.number_input("Perimeter Mean", value=122.8)
area_mean = st.sidebar.number_input("Area Mean", value=1001.0)
smoothness_mean = st.sidebar.number_input("Smoothness Mean", value=0.1184)
compactness_mean = st.sidebar.number_input("Compactness Mean", value=0.22862)
concavity_mean = st.sidebar.number_input("Concavity Mean", value=0.28241)
concave_points_mean = st.sidebar.number_input("Concave Points Mean", value=0.1471)
symmetry_mean = st.sidebar.number_input("Symmetry Mean", value=0.2419)
fractal_dimension_mean = st.sidebar.number_input("Fractal Dimension Mean", value=0.07871)

# Collect input features into a list
user_input = np.array([[
    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
    fractal_dimension_mean
]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(user_input)
    result = "Malignant" if prediction[0] == 1 else "Benign"
    st.write(f"The predicted diagnosis is: **{result}**")

# Footer
st.markdown("---")
st.markdown("Developed by [Your Name]")
