import os

class Config:
    """Simple application configuration"""
    DEBUG = True
    SECRET_KEY = 'your-secret-key-here'
    MODEL_PATH = 'models/diabetes_model.pkl'
    SCALER_PATH = 'models/scaler.pkl'
    MEDIAN_VALUES_PATH = 'models/median_values.pkl'
    
    # Feature validation ranges
    FEATURE_LIMITS = {
        'pregnancies': {'min': 0, 'max': 20},
        'glucose': {'min': 0, 'max': 300},
        'blood_pressure': {'min': 0, 'max': 200},
        'bmi': {'min': 10.0, 'max': 70.0},
        'age': {'min': 10, 'max': 120},
        'insulin': {'min': 0, 'max': 900}
    }

config = Config()
