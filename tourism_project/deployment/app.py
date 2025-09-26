import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the saved model from the Hugging Face model hub
model_path = hf_hub_download(repo_id="adityasharma0511/predict-model", filename="best_predict_model.joblib")

# Load the saved model from the Hugging Face model hub
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Customer Package Purchase prediction App")
st.write("Customer Package Purchase prediction App is an internal tool for Visit_with_Us that predicts whether customers will buy a package or not.")
st.write("Kindly enter the customer details to check whether they are likely to buy package.")

# Get the inputs and save them into a dataframe
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
CityTier = st.number_input("City Tier (customer's CityTier)", min_value=1, max_value=5, value=3)
DurationOfPitch = st.number_input("Duration Of Pitch (DurationOfPitch to customer)", min_value=0, max_value=100, value=50)
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=0, max_value=50, value=1)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, max_value=50, value=1)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
NumberOfTrips = st.number_input("Number Of Trips", min_value=0, max_value=100, value=1)
Passport = st.number_input("Passport (1 = Yes, 0 = No)", min_value=0, max_value=1, value=0)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=0, max_value=10, value=5)
OwnCar = st.number_input("Own Car (1 = Yes, 0 = No)", min_value=0, max_value=1, value=0)
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=20, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=50000)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Female", "Male", "Fe Male"])
ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])

# Save the inputs into a Dataframe. Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Purchase" if prediction == 1 else "not purchase"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
