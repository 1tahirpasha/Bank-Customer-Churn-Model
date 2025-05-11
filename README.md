# 🏦 Bank Customer Churn Prediction App

<a href="https://bank-customer-churn-model-by-1tahirpasha.streamlit.app/">
    <img src="https://img.shields.io/badge/Streamlit-Cloud-green" width="300" />
</a>

<a href="https://bank-customer-churn-model.onrender.com">
    <img src="https://img.shields.io/badge/Render-Deployed-lightgrey" width="300" />
</a>

This Streamlit-based app predicts whether a bank customer is likely to churn using a machine learning model trained on customer demographics, account activity, and product usage patterns.

## 🔗 Links :  

- 🌐 [Streamlit Cloud (Fast Load)](https://bank-customer-churn-model-by-1tahirpasha.streamlit.app)
- 💤 [Render (May Sleep)](https://bank-customer-churn-model.onrender.com)

---

## ✨ Features

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
<img src="https://github.com/1tahirpasha/Bank-Customer-Churn-Model/blob/main/app_preview.png?raw=true" width="700"/>

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
