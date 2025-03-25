from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score
import joblib
import os

# Load all models at startup
models = {
    'simple_linear': joblib.load('models/simple_linear_model.pkl'),
    'multiple_linear': joblib.load('models/multiple_linear_model.pkl'),
    'polynomial': {
        'poly_features': joblib.load('models/poly_features.pkl'),
        'model': joblib.load('models/polynomial_model.pkl')
    },
    'logistic': joblib.load('models/logistic_model.pkl'),
    'knn': joblib.load('models/knn_model.pkl')
}

# Model accuracies
accuracies = {
    'simple_linear': 0.9398,
    'multiple_linear': 0.9917,
    'polynomial': 0.9933,
    'logistic': 1.0000,
    'knn': 0.9000
}

# Input field descriptions and normal ranges
input_descriptions = {
    'simple_linear': {
        'vehicle_count': {
            'description': 'Number of vehicles passing through the area per hour',
            'normal_range': '50-500 vehicles/hour'
        }
    },
    'multiple_linear': {
        'age': {
            'description': 'Age of the patient in years',
            'normal_range': '18-80 years'
        },
        'cholesterol': {
            'description': 'Total cholesterol level in mg/dL',
            'normal_range': '125-200 mg/dL'
        },
        'blood_pressure': {
            'description': 'Systolic blood pressure in mmHg',
            'normal_range': '90-140 mmHg'
        }
    },
    'polynomial': {
        'diet_score': {
            'description': 'Diet quality score (0-10)',
            'normal_range': '0-10 points'
        },
        'exercise_duration': {
            'description': 'Daily exercise duration in minutes',
            'normal_range': '15-60 minutes'
        }
    },
    'logistic': {
        'income': {
            'description': 'Annual income in dollars',
            'normal_range': '$30,000-$150,000'
        },
        'credit_score': {
            'description': 'Credit score',
            'normal_range': '300-850'
        },
        'loan_amount': {
            'description': 'Requested loan amount in dollars',
            'normal_range': '$5,000-$500,000'
        }
    },
    'knn': {
        'packet_size': {
            'description': 'Network packet size in bytes',
            'normal_range': '64-1518 bytes'
        },
        'connection_duration': {
            'description': 'Connection duration in seconds',
            'normal_range': '0-300 seconds'
        },
        'num_failed_attempts': {
            'description': 'Number of failed connection attempts',
            'normal_range': '0-20 attempts'
        },
        'source_bytes': {
            'description': 'Bytes sent from source',
            'normal_range': '0-25000 bytes'
        },
        'destination_bytes': {
            'description': 'Bytes received by destination',
            'normal_range': '0-5000 bytes'
        }
    }
}

def home(request):
    return render(request, 'home.html', {'accuracies': accuracies})

def simple_linear(request):
    result = None
    
    if request.method == 'POST':
        vehicle_count = float(request.POST.get('vehicle_count'))
        prediction = models['simple_linear'].predict([[vehicle_count]])[0]
        result = {'prediction': round(prediction, 2)}
    
    context = {
        'result': result,
        'accuracy': accuracies['simple_linear'],
        'input_info': input_descriptions['simple_linear']
    }
    return render(request, 'simple_linear.html', context)

def multiple_linear(request):
    result = None
    
    if request.method == 'POST':
        age = float(request.POST.get('age'))
        cholesterol = float(request.POST.get('cholesterol'))
        blood_pressure = float(request.POST.get('blood_pressure'))
        prediction = models['multiple_linear'].predict([[age, cholesterol, blood_pressure]])[0]
        result = {'prediction': round(prediction, 2)}
    
    context = {
        'result': result,
        'accuracy': accuracies['multiple_linear'],
        'input_info': input_descriptions['multiple_linear']
    }
    return render(request, 'multiple_linear.html', context)

def polynomial(request):
    result = None
    
    if request.method == 'POST':
        diet_score = float(request.POST.get('diet_score'))
        exercise_duration = float(request.POST.get('exercise_duration'))
        X_poly = models['polynomial']['poly_features'].transform([[diet_score, exercise_duration]])
        prediction = models['polynomial']['model'].predict(X_poly)[0]
        result = {'prediction': round(prediction, 2)}
    
    context = {
        'result': result,
        'accuracy': accuracies['polynomial'],
        'input_info': input_descriptions['polynomial']
    }
    return render(request, 'polynomial.html', context)

def logistic(request):
    result = None
    
    if request.method == 'POST':
        income = float(request.POST.get('income'))
        credit_score = float(request.POST.get('credit_score'))
        loan_amount = float(request.POST.get('loan_amount'))
        prediction = models['logistic'].predict([[income, credit_score, loan_amount]])[0]
        result = {
            'prediction': 'Approved' if prediction == 1 else 'Not Approved'
        }
    
    context = {
        'result': result,
        'accuracy': accuracies['logistic'],
        'input_info': input_descriptions['logistic']
    }
    return render(request, 'logistic.html', context)

def knn(request):
    result = None
    
    if request.method == 'POST':
        packet_size = float(request.POST.get('packet_size'))
        connection_duration = float(request.POST.get('connection_duration'))
        num_failed_attempts = float(request.POST.get('num_failed_attempts'))
        source_bytes = float(request.POST.get('source_bytes'))
        destination_bytes = float(request.POST.get('destination_bytes'))
        
        prediction = models['knn'].predict([[packet_size, connection_duration, num_failed_attempts, source_bytes, destination_bytes]])[0]
        
        attack_types = {
            0: 'Normal Traffic',
            1: 'DDoS Attack',
            2: 'Port Scanning',
            3: 'Brute Force'
        }
        
        result = {
            'prediction': attack_types[prediction]
        }
    
    context = {
        'result': result,
        'accuracy': accuracies['knn'],
        'input_info': input_descriptions['knn']
    }
    return render(request, 'knn.html', context)
