import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os

print("Training Simple Linear Regression Model for Air Pollution...")

# Load and prepare data
df = pd.read_csv('air_pollution_dataset.csv')
X = df[['Vehicle_Count']]
y = df['Pollution_Level_AQI']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Test accuracy
accuracy = r2_score(y_test, model.predict(X_test))
print(f"Simple Linear Regression (Air Pollution) Accuracy: {accuracy:.4f}")

# Save the model
joblib.dump(model, 'models/simple_linear_model.pkl')
print("Model saved as simple_linear_model.pkl")
