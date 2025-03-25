import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os

print("Training Multiple Linear Regression Model for Heart Disease Risk...")

# Load and prepare data
df = pd.read_csv('heart_disease_risk_dataset.csv')
X = df[['Age', 'Cholesterol', 'Blood_Pressure']]
y = df['Heart_Disease_Risk']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Test accuracy
accuracy = r2_score(y_test, model.predict(X_test))
print(f"Multiple Linear Regression (Heart Disease) Accuracy: {accuracy:.4f}")

# Save the model
joblib.dump(model, 'models/multiple_linear_model.pkl')
print("Model saved as multiple_linear_model.pkl")
