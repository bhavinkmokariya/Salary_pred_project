import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('Salary_Prediction_Model.joblib')

# Set up the Streamlit app
st.title("Salary Range Prediction App")
st.write("Predict the salary range for a job posting based on its characteristics.")

# Input fields for the user
job_title = st.text_input("Job Title")
job_category = st.selectbox("Job Category", options=["Technology", "Healthcare", "Finance", "Education", "Legal", "Engineering"])
career_level = st.selectbox("Career Level", options=["Entry", "Mid", "Senior"])
employment_type = st.selectbox("Employment Type", options=["Full-Time", "Part-Time"])
job_location = st.text_input("Job Location (e.g., New York)")

# Numerical inputs
experience_years = st.number_input("Years of Experience Required", min_value=0, max_value=50, value=0)
min_salary = st.number_input("Minimum Salary Offer ($)", min_value=0, value=0)
max_salary = st.number_input("Maximum Salary Offer ($)", min_value=0, value=0)

# Encode categorical inputs (this encoding should match what was used during model training)
job_category_mapping = {"Technology": 0, "Healthcare": 1, "Finance": 2, "Education": 3, "Legal": 4, "Engineering": 5}
career_level_mapping = {"Entry": 0, "Mid": 1, "Senior": 2}
employment_type_mapping = {"Full-Time": 0, "Part-Time": 1}

job_category_encoded = job_category_mapping[job_category]
career_level_encoded = career_level_mapping[career_level]
employment_type_encoded = employment_type_mapping[employment_type]

# Button to make prediction
if st.button("Predict Salary Range"):
    # Create a feature array for prediction
    features = np.array([[job_category_encoded, career_level_encoded, employment_type_encoded, experience_years, min_salary, max_salary]])
    
    # Make a prediction
    predicted_salary = model.predict(features)

    # Display the result
    st.write(f"Predicted Salary Midpoint: ${predicted_salary[0]:,.2f}")
