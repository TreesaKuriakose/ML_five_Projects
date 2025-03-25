import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score

def test_multiple_linear_new():
    df = pd.read_csv('heart_disease_risk_dataset_new.csv')
    X = df[['Age', 'Cholesterol', 'Blood_Pressure']]
    y = df['Heart_Disease_Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    accuracy = r2_score(y_test, model.predict(X_test))
    print(f"Multiple Linear Regression (Heart Disease) New Accuracy: {accuracy:.4f}")
    return accuracy

def test_logistic_new():
    df = pd.read_csv('loan_approval_dataset_new.csv')
    X = df[['Income', 'Credit_Score', 'Loan_Amount']]
    y = df['Loan_Approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Logistic Regression (Loan Approval) New Accuracy: {accuracy:.4f}")
    return accuracy

def test_knn_new():
    df = pd.read_csv('cyber_attack_classification_dataset_new.csv')
    X = df[['Packet_Size', 'Connection_Duration', 'Num_Failed_Attempts', 'Source_Bytes', 'Destination_Bytes']]
    y = df['Attack_Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"KNN (Cyber Attack) New Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    print("\nTesting new datasets accuracies...")
    print("-" * 50)
    accuracies = {
        "Multiple Linear": test_multiple_linear_new(),
        "Logistic": test_logistic_new(),
        "KNN": test_knn_new()
    }
    
    print("\nNew dataset accuracies summary:")
    print("-" * 50)
    for model, acc in accuracies.items():
        print(f"{model}: New accuracy = {acc:.4f}")
