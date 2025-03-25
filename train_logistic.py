import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

print("Training Logistic Regression Model for Loan Approval...")

# Load and prepare data
df = pd.read_csv('loan_approval_dataset.csv')
X = df[['Income', 'Credit_Score', 'Loan_Amount']]
y = df['Loan_Approved']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Logistic Regression (Loan Approval) Accuracy: {accuracy:.4f}")

# Save the model
joblib.dump(model, 'models/logistic_model.pkl')
print("Model saved as logistic_model.pkl")
