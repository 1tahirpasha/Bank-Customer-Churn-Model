# IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read Dataset
df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Bank%20Churn%20Modelling.csv')
df = df.set_index('CustomerId')

# Preprocessing
df.replace({'Geography': {'France': 2, 'Germany': 1, 'Spain': 0}}, inplace=True)
df.replace({'Gender': {'Male': 0, 'Female': 1}}, inplace=True)
df.replace({'Num Of Products': {1: 0, 2: 1, 3: 1, 4: 1}}, inplace=True)
df['Zero Balance'] = np.where(df['Balance'] > 0, 1, 0)

# Features and target
X = df.drop(['Surname', 'Churn'], axis=1)
y = df['Churn']

# Handle class imbalance
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=2529)
X_rus, y_rus = rus.fit_resample(X, y)

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=2529)
X_ros, y_ros = ros.fit_resample(X, y)

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2529)
X_train_rus, X_test_rus, y_train_rus, y_test_rus = train_test_split(X_rus, y_rus, test_size=0.3, random_state=2529)
X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(X_ros, y_ros, test_size=0.3, random_state=2529)

# Standardize Features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train[['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']] = sc.fit_transform(X_train[['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']])
X_test[['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']] = sc.transform(X_test[['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']])

X_train_rus[['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']] = sc.fit_transform(X_train_rus[['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']])
X_test_rus[['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']] = sc.transform(X_test_rus[['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']])

X_train_ros[['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']] = sc.fit_transform(X_train_ros[['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']])
X_test_ros[['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']] = sc.transform(X_test_ros[['CreditScore', 'Age', 'Tenure', 'Balance', 'Estimated Salary']])

# Support Vector Machine Classifier
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

# Evaluation
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf'],
    'class_weight': ['balanced']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=2)
grid.fit(X_train, y_train)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
confusion_matrix(y_test, grid_predictions)
print(classification_report(y_test, grid_predictions))

# Under Sampling Model
svc_rus = SVC()
svc_rus.fit(X_train_rus, y_train_rus)
y_pred_rus = svc_rus.predict(X_test_rus)
confusion_matrix(y_test_rus, y_pred_rus)
print(classification_report(y_test_rus, y_pred_rus))

# Under Sampling Tuning
grid_rus = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=2)
grid_rus.fit(X_train_rus, y_train_rus)
print(grid_rus.best_estimator_)
grid_predictions_rus = grid_rus.predict(X_test_rus)
confusion_matrix(y_test_rus, grid_predictions_rus)
print(classification_report(y_test_rus, y_pred_rus))

# Over Sampling Model
svc_ros = SVC()
svc_ros.fit(X_train_ros, y_train_ros)
y_pred_ros = svc_ros.predict(X_test_ros)
confusion_matrix(y_test_ros, y_pred_ros)
print(classification_report(y_test_ros, y_pred_ros))

# Over Sampling Tuning
grid_ros = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=2)
grid_ros.fit(X_train_ros, y_train_ros)
print(grid_ros.best_estimator_)
grid_predictions_ros = grid_ros.predict(X_test_ros)
confusion_matrix(y_test_ros, grid_predictions_ros)
print(classification_report(y_test_ros, grid_predictions_ros))

# Train a Random Forest model on dummy data (optional)
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model to a pickle file
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Download model (for Colab)
from google.colab import files
files.download("model.pkl")
