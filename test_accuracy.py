import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score

def test_simple_linear():
    df = pd.read_csv('air_pollution_dataset.csv')
    X = df[['Vehicle_Count']]
    y = df['Pollution_Level_AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    accuracy = r2_score(y_test, model.predict(X_test))
    print(f"Simple Linear Regression (Air Pollution) Accuracy: {accuracy:.4f}")
    return accuracy

def test_multiple_linear():
    df = pd.read_csv('heart_disease_risk_dataset.csv')
    X = df[['Age', 'Cholesterol', 'Blood_Pressure']]
    y = df['Heart_Disease_Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    accuracy = r2_score(y_test, model.predict(X_test))
    print(f"Multiple Linear Regression (Heart Disease) Accuracy: {accuracy:.4f}")
    return accuracy

def test_polynomial():
    df = pd.read_csv('blood_sugar_prediction_dataset.csv')
    X = df[['Diet_Score', 'Exercise_Duration']]
    y = df['Blood_Sugar_Level']
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    accuracy = r2_score(y_test, model.predict(X_test))
    print(f"Polynomial Regression (Blood Sugar) Accuracy: {accuracy:.4f}")
    return accuracy

def test_logistic():
    df = pd.read_csv('loan_approval_dataset.csv')
    X = df[['Income', 'Credit_Score', 'Loan_Amount']]
    y = df['Loan_Approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Logistic Regression (Loan Approval) Accuracy: {accuracy:.4f}")
    return accuracy

def test_knn():
    df = pd.read_csv('cyber_attack_classification_dataset.csv')
    X = df[['Packet_Size', 'Connection_Duration', 'Num_Failed_Attempts', 'Source_Bytes', 'Destination_Bytes']]
    y = df['Attack_Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"KNN (Cyber Attack) Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    print("\nTesting model accuracies...")
    print("-" * 50)
    accuracies = {
        "Simple Linear": test_simple_linear(),
        "Multiple Linear": test_multiple_linear(),
        "Polynomial": test_polynomial(),
        "Logistic": test_logistic(),
        "KNN": test_knn()
    }
    
    print("\nDatasets needing replacement (accuracy < 0.9):")
    print("-" * 50)
    for model, acc in accuracies.items():
        if acc < 0.9:
            print(f"{model}: Current accuracy = {acc:.4f}")
