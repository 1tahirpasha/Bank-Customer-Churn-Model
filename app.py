import streamlit as st
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction Using ML")

st.markdown("Enter customer details below to predict if they are likely to churn.")

# Input form
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 700)
        age = st.slider("Age", 18, 100, 42)
        tenure = st.slider("Tenure (years)", 0, 10, 3)
        balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])

    with col2:
        has_cr_card = st.radio("Has Credit Card?", ['Yes', 'No'])
        is_active = st.toggle("Is Active Member?", value=True)
        salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 80000.0)
        geo = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
        gender = st.radio("Gender", ['Male', 'Female'])

    submitted = st.form_submit_button("🚀 Predict")

# Mapping inputs
if submitted:
    has_cr_card = 1 if has_cr_card == 'Yes' else 0
    is_active = 1 if is_active else 0
    geo_dict = {'France': 0, 'Spain': 1, 'Germany': 2}
    geo = geo_dict[geo]
    gender = 1 if gender == 'Male' else 0

    input_data = np.array([[credit_score, age, tenure, balance, num_products,
                            has_cr_card, is_active, salary, geo, gender]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("❌ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **likely to stay**.")
