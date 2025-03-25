import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import joblib
import os

print("Training Polynomial Regression Model for Blood Sugar...")

# Load and prepare data
df = pd.read_csv('blood_sugar_prediction_dataset.csv')
X = df[['Diet_Score', 'Exercise_Duration']]
y = df['Blood_Sugar_Level']

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Test accuracy
accuracy = r2_score(y_test, model.predict(X_test))
print(f"Polynomial Regression (Blood Sugar) Accuracy: {accuracy:.4f}")

# Save both the polynomial features transformer and the model
joblib.dump(poly, 'models/poly_features.pkl')
joblib.dump(model, 'models/polynomial_model.pkl')
print("Models saved as poly_features.pkl and polynomial_model.pkl")
