from flask import Flask, request, render_template, send_file, jsonify
import os
from datetime import datetime

# Import our modules
from config import config
from validators import validate_form_data
from model_predictor import get_predictor
from report_generator import ReportGenerator

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY

# Initialize components
try:
    predictor = get_predictor()
    report_generator = ReportGenerator()
    print("Application initialized successfully")
except Exception as e:
    print(f"Error initializing application: {e}")
    print("Please make sure the model is trained. Run: python model_trainer.py")

@app.route('/')
def form():
    """Display the main form"""
    return render_template('form.html')

@app.route('/', methods=['POST'])
def handle_form_submission():
    """Handle form submission and generate report"""
    try:
        # Get form data
        form_data = {
            'pregnancies': request.form.get('pregnancies'),
            'glucose': request.form.get('glucose'),
            'blood_pressure': request.form.get('blood_pressure'),
            'bmi': request.form.get('bmi'),
            'age': request.form.get('age'),
            'insulin': request.form.get('insulin')
        }
        
        # Validate input
        is_valid, errors = validate_form_data(form_data)
        if not is_valid:
            return jsonify({'errors': errors}), 400
        
        # Convert to appropriate types
        pregnancies = int(form_data['pregnancies'])
        glucose = int(form_data['glucose'])
        blood_pressure = int(form_data['blood_pressure'])
        bmi = float(form_data['bmi'])
        age = int(form_data['age'])
        insulin = int(form_data['insulin'])
        
        # Create input data
        input_data = (pregnancies, glucose, blood_pressure, insulin, bmi, age)
        
        # Create input dictionary for reporting
        input_data_dict = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'Insulin': insulin,
            'BMI': bmi,
            'Age': age
        }
        
        # Make prediction
        prediction, probability = predictor.predict(input_data)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f'diabetes_report_{timestamp}.pdf'
        
        # Ensure reports directory exists
        os.makedirs('reports', exist_ok=True)
        report_path = os.path.join('reports', report_path)
        
        report_generator.generate_report(input_data_dict, prediction, probability, report_path)
        
        # Return PDF file
        return send_file(report_path, as_attachment=True, download_name='diabetes_report.pdf')
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    # Create required directories
    os.makedirs('reports', exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(config.MODEL_PATH):
        print("Warning: Model not found. Please train the model first by running: python model_trainer.py")
    
    app.run(debug=config.DEBUG, host='0.0.0.0', port=5000)
