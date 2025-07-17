import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from app import app
from validators import validate_form_data
from model_predictor import DiabetesPredictor
from config import config

def test_form_validation():
    """Test form validation"""
    # Valid data
    form_data = {
        'pregnancies': '1',
        'glucose': '100',
        'blood_pressure': '70',
        'bmi': '25.0',
        'age': '30',
        'insulin': '100'
    }
    
    is_valid, errors = validate_form_data(form_data)
    assert is_valid == True
    assert len(errors) == 0
    print("✓ Form validation test passed")

def test_invalid_data():
    """Test form validation with invalid data"""
    form_data = {
        'pregnancies': '-1',  # Invalid
        'glucose': '400',     # Invalid
        'blood_pressure': '70',
        'bmi': '25.0',
        'age': '30',
        'insulin': '100'
    }
    
    is_valid, errors = validate_form_data(form_data)
    assert is_valid == False
    assert len(errors) > 0
    print("✓ Invalid data test passed")

def test_model_prediction():
    """Test model prediction if model exists"""
    if os.path.exists(config.MODEL_PATH):
        predictor = DiabetesPredictor()
        test_input = (1, 100, 70, 100, 25.0, 30)
        prediction, probability = predictor.predict(test_input)
        
        assert prediction in [0, 1]
        assert 0 <= probability <= 1
        print("✓ Model prediction test passed")
    else:
        print("⚠ Model not found, skipping prediction test")

def test_app_routes():
    """Test basic app routes"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        # Test form page
        response = client.get('/')
        assert response.status_code == 200
        print("✓ Form page test passed")
        
        # Test health check
        response = client.get('/health')
        assert response.status_code == 200
        print("✓ Health check test passed")

def run_tests():
    """Run all tests"""
    print("Running simplified diabetes prediction tests...")
    
    try:
        test_form_validation()
        test_invalid_data()
        test_model_prediction()
        test_app_routes()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == '__main__':
    run_tests()
