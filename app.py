import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Lottie animation function
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Sidebar animation
with st.sidebar:
    st.title("📊 About This App")
    st.info("This app predicts if a bank customer will churn based on key factors.")
    lottie = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_w51pcehl.json")
    if lottie:
        st_lottie(lottie, height=180)

# Main Title
st.markdown("## 📈 Customer Churn Prediction Using ML")
st.caption("Enter customer details below to predict if they are likely to churn.")

# Input fields
credit_score = st.slider("Credit Score", 300, 900, 700)
age = st.slider("Age", 18, 100, 42)
tenure = st.slider("Tenure (years)", 0, 10, 3)
balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.radio("Has Credit Card?", ['Yes', 'No']) == 'Yes'
is_active = st.toggle("Is Active Member?", value=True)
salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 80000.0)
geo = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
gender = st.radio("Gender", ['Male', 'Female'])

# Encode values
geo_dict = {'France': 0, 'Spain': 1, 'Germany': 2}
geo = geo_dict[geo]
gender = 1 if gender == 'Male' else 0
has_cr_card = int(has_cr_card)
is_active = int(is_active)

# Prediction
if st.button("🚀 Predict"):
    user_inputs = [credit_score, age, tenure, balance, num_products, has_cr_card,
                   is_active, salary, geo, gender]
    input_data = np.array([user_inputs])
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]

    result = "🛑 Customer will churn" if prediction[0] == 1 else "✅ Customer will stay"
    st.success(result)
    st.info(f"🔎 Confidence: {proba*100:.2f}% chance of churn")

# Prepare download data
data = f"Prediction: {result_text}\nConfidence: {confidence:.2f}% chance of churn"

# Add download button
st.download_button("📄 Download Result", data, file_name="prediction_result.txt")

# Feature Importance
with st.expander("📊 Show Feature Importance"):
    importances = model.feature_importances_
    features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
                "IsActiveMember", "EstimatedSalary", "Geography", "Gender"]
    fig, ax = plt.subplots()
    ax.barh(features, importances, color='skyblue')
    ax.set_title("Feature Importance")
    st.pyplot(fig)
