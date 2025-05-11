%%writefile app.py
import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction Using ML")

# Input fields
credit_score = st.number_input("CreditScore", 300, 900, 700)
age = st.number_input("Age", 18, 100, 42)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.number_input("NumOfProducts", 1, 4, 1)
has_cr_card = st.selectbox("HasCrCard", ['yes', 'no'])
is_active = st.selectbox("IsActiveMember", ['yes', 'no'])
salary = st.number_input("EstimatedSalary", 0.0, 200000.0, 80000.0)
geo = st.selectbox("Geography", ['France', 'Spain', 'Germany'])  # You can encode these as 0,1,2
gender = st.selectbox("Gender", ['Male', 'Female'])

# Map inputs
has_cr_card = 1 if has_cr_card == 'yes' else 0
is_active = 1 if is_active == 'yes' else 0
geo_dict = {'France': 0, 'Spain': 1, 'Germany': 2}
geo = geo_dict[geo]
gender = 1 if gender == 'Male' else 0

# Prediction
if st.button("Predict"):
    user_inputs = [credit_score, age, tenure, balance, num_products, has_cr_card, is_active, salary, geo, gender]
    input_data = np.array([user_inputs]).reshape(1, -1)
    prediction = model.predict(input_data)
    result = "Customer will churn" if prediction[0] == 1 else "Customer will stay"
    st.success(result)
