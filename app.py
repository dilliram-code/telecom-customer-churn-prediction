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


X_transformed = preprocessor.transform(input_data)
feature_names =  preprocessor.get_feature_names_out()
X_transformed_df = pd.DataFrame(
  X_transformed,
  columns = feature_names
)

explainer = shap.TreeExplainer(rf_classifier)

# make prediction based on model choice
if model_choice == "Logistic Regression":
  prediction = logistic_model.predict_proba(input_data)[0][1]
  # st.write(f"Churn Probability (Logistic Regression): {prediction:.2f}")
elif model_choice == "Random Forest":
  prediction = rf_model.predict_proba(input_data)[0][1]
  # st.write(f"Churn Probability (Random Forest): {prediction:.2f}")
else:
  prediction = xgb_model.predict_proba(input_data)[0][1]
  # st.write(f"Churn Probability (XGBoost): {prediction:.2f}")
  
if st.button("Predict Churn"):
  # st.write(f"Churn Probability ({model_choice}): {prediction:.2f}")
  prob =  prediction
  st.metric("Churn Probability ", f"{prob * 100: .1f} %")
  st.progress(float(prob))
  
  if prob > 0.6:
    st.error("High Risk Customer")
  elif prob > 0.3:
    st.warning("Medium Risk Customer")
  else:
    st.success("Low Risk Customer")
  
  st.write(f"Model Used: ** {model_choice}")
  
  
  st.subheader("Prediction Explanation (SHAP)")
  X_transformed = preprocessor.transform(input_data)
  explainer = shap.Explainer(rf_classifier)
  shap_values = explainer(X_transformed_df)
  
  fig = plt.figure()
  
  shap.plots.waterfall(
    shap_values[0, :, 1],
    show = False
  )
  st.pyplot(fig)
  


# code for tab2
with tab2:
  st.title("Model Insights")
  st.write("Feature importance and insights from the model")
  
  preprocessor = rf_model.named_steps['preprocessor']
  feature_names = preprocessor.get_feature_names_out()
  
  rf_classifier = rf_model.named_steps['model']
  rf_importance = rf_classifier.feature_importances_
  
  importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importance
  }).sort_values(by='Importance', ascending=False)
  
  top_features = importance_df.head(15)
  
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.barh(top_features['Feature'], top_features['Importance'])
  ax.set_xlabel("Feature Importance")
  ax.set_title("Top Drivers of Customer churn")
  plt.gca().invert_yaxis()
  st.pyplot(fig)
  
  