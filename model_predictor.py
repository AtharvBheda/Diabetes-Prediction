import joblib
import numpy as np
import os
from config import config

class DiabetesPredictor:
    def __init__(self):
        self.model = joblib.load(config.MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        self.median_values = joblib.load(config.MEDIAN_VALUES_PATH)
        self.features = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']
    
    def replace_zero_with_median(self, input_data):
        """Replace zero values with median values"""
        input_dict = dict(zip(self.features, input_data))
        
        for feature in self.features:
            if input_dict[feature] == 0:
                input_dict[feature] = self.median_values[feature]
        
        return np.array(list(input_dict.values()))
    
    def predict(self, input_data):
        """Make diabetes prediction"""
        # Replace zeros with median values
        input_data_cleaned = self.replace_zero_with_median(input_data)
        
        # Scale the input data
        input_data_scaled = self.scaler.transform([input_data_cleaned])
        
        # Make prediction
        prediction = self.model.predict(input_data_scaled)[0]
        probability = self.model.predict_proba(input_data_scaled)[0][1]
        
        return int(prediction), float(probability)

# Global predictor instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = DiabetesPredictor()
    return predictor
