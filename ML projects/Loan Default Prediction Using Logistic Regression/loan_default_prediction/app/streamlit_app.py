import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('models/logistic_model.pkl')
scaler = StandardScaler()

st.title("Loan Default Risk Predictor")

# Input features
st.subheader("Personal Information")
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

st.subheader("Financial Information")
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0)
credit_history = st.selectbox("Credit History", [0, 1], help="0 = No credit history, 1 = Has credit history")
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

if st.button("Predict"):
    # Create feature dictionary
    features = {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Married_Yes': 1 if married == "Yes" else 0,
        'Dependents_1': 1 if dependents == "1" else 0,
        'Dependents_2': 1 if dependents == "2" else 0,
        'Dependents_3+': 1 if dependents == "3+" else 0,
        'Education_Not Graduate': 1 if education == "Not Graduate" else 0,
        'Self_Employed_Yes': 1 if self_employed == "Yes" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0
    }
    
    # Convert to array and scale
    feature_array = np.array(list(features.values())).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(feature_array)[0]
    
    # Show prediction
    if prediction == 1:
        st.error("Prediction: High Risk of Default")
    else:
        st.success("Prediction: Low Risk of Default")
        
    # Show probability
    proba = model.predict_proba(feature_array)[0]
    st.info(f"Probability of Default: {proba[1]:.2%}")