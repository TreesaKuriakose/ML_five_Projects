import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("Training KNN Model for Cyber Attack Classification...")

# Load and prepare data
df = pd.read_csv('cyber_attack_classification_dataset.csv')
X = df[['Packet_Size', 'Connection_Duration', 'Num_Failed_Attempts', 'Source_Bytes', 'Destination_Bytes']]
y = df['Attack_Type']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Test accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"KNN (Cyber Attack) Accuracy: {accuracy:.4f}")

# Save the model
joblib.dump(model, 'models/knn_model.pkl')
print("Model saved as knn_model.pkl")
