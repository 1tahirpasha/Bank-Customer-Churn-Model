# 🏦 Bank Customer Churn Prediction App

[![Render](https://github.com/user-attachments/assets/b7acdc15-5a4c-48bd-9220-4a4a1b97be12)](https://bank-customer-churn-model.onrender.com)

This Streamlit-based app predicts whether a bank customer is likely to churn using a machine learning model trained on customer demographics, account activity, and product usage patterns.

🔗 **Live Demo**: [Click to Open the App](https://bank-customer-churn-model.onrender.com)

---

## 🧠 Features

- Predicts customer churn based on:
  - Credit Score
  - Age, Balance, Tenure
  - Geography, Gender
  - Products, Credit Card, Salary, and more
- Trained using a Random Forest Classifier
- Simple web UI with sliders & dropdowns
- Easy to use on desktop or mobile
- Optionally extendable with SHAP for explainability

---

## 📷 App Preview
![image](https://github.com/user-attachments/assets/0a3f64b8-2a0f-4371-b901-1ff45b54cf03)

---

## 🚀 How to Run Locally

```bash
# Clone this repo
git clone https://github.com/yourusername/Bank-Customer-Churn-Model.git
cd Bank-Customer-Churn-Model

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
