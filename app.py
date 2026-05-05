import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import shap 


# load the trained models
logistic_model = load("models/logistic_model.pkl")
rf_model = load("models/rf_model.pkl")
xgb_model = load("models/xgb_model.pkl")


# make interface of the app
tab1, tab2 = st.tabs(['Prediction', 'Model insights'])
preprocessor = rf_model.named_steps['preprocessor']
rf_classifier = rf_model.named_steps['model']


# code for tab1
with tab1:
  st.title("Customer Churn Prediction App")
  st.write("Enter customer details to predict churn probability")

  # drop down for model selection
  model_choice = st.selectbox(
    "Select a model",
    ("Logistic Regression", "Random Forest", "XGBoost")
  )
  
  # input fields for customer data
  gender = st.selectbox("Gender", ['Male', 'Female'])
  senior_citizen = st.selectbox("Senior Citizen", [0, 1])
  partner = st.selectbox("Partner", ['Yes', 'No'])
  dependents = st.selectbox("Dependents", ['Yes', 'No'])
  tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
  phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
  multiple_lines = st.selectbox("Multiple Lines", ['No phone service', 'Yes',
                                                    'No'])
  internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic',
                                                        'No'])
  online_security = st.selectbox("Online Security", ['No internet service',
                                                        'Yes', 'No'])
  online_backup = st.selectbox("Online Backup", ['No internet service',
                                                        'Yes', 'No'])
  device_protection = st.selectbox("Device Protection", ['No internet service',
                                                        'Yes', 'No'])
  tech_support = st.selectbox("Tech Support", ['No internet service',
                                                        'Yes', 'No'])
  streaming_tv = st.selectbox("Streaming TV", ['No internet service',
                                                        'Yes', 'No'])
  streaming_movies = st.selectbox("Streaming Movies", ['No internet service',
                                                        'Yes', 'No'])
  contract = st.selectbox("Contract", ['Month-to-month', 'One year',
                                                        'Two year'])
  paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
  payment_method = st.selectbox("Payment Method", ['Electronic check',
                                                        'Mailed check', 'Bank transfer (automatic)',
                                                        'Credit card (automatic)'])
  monthly_charges = st.number_input("Monthly Charges", min_value=0.0,
                                      max_value=1000.0, value=70.0)
  total_charges = st.number_input("Total Charges", min_value=0.0,
                                      max_value=10000.0, value=2000.0)
  
  
  # create a dataframe from the input data
  input_data = pd.DataFrame({
      'gender': [gender], 
      'SeniorCitizen': [senior_citizen], 
      'Partner': [partner], 
      'Dependents': [dependents],
      'tenure': [tenure], 
      'PhoneService': [phone_service], 
      'MultipleLines': [multiple_lines], 
      'InternetService': [internet_service],
      'OnlineSecurity': [online_security], 
      'OnlineBackup': [online_backup], 
      'DeviceProtection': [device_protection], 
      'TechSupport': [tech_support],
      'StreamingTV': [streaming_tv], 
      'StreamingMovies': [streaming_movies], 
      'Contract': [contract], 
      'PaperlessBilling': [paperless_billing],
      'PaymentMethod': [payment_method], 
      'MonthlyCharges': [monthly_charges], 
      'TotalCharges': [total_charges]
  })
  
  