from flask import Flask, request, render_template, send_file
from diabetes_model import generate_report, predict_diabetes, replace_zero_with_median
import os

app = Flask(__name__)

# Route for the form
@app.route('/')
def form():
    return render_template('form.html')  # Renders the form for user input

# Route to handle form submission
@app.route('/', methods=['POST'])
def handle_form_submission():
    # Get the input data from the form
    pregnancies = int(request.form.get('pregnancies'))
    glucose = int(request.form.get('glucose'))
    blood_pressure = int(request.form.get('blood_pressure'))
    bmi = float(request.form.get('bmi'))
    age = int(request.form.get('age'))
    insulin = int(request.form.get('insulin'))  # Insulin input from user

    # Create input_data tuple for prediction (include relevant columns except SkinThickness, DiabetesPedigreeFunction)
    input_data = (pregnancies, glucose, blood_pressure, insulin, bmi, age)

    # Replace 0 values with the median from the dataset
    input_data_corrected = replace_zero_with_median(input_data)

    # Make the diabetes prediction
    prediction = predict_diabetes(input_data_corrected)

    # Generate the PDF report
    report_path = generate_report(input_data_corrected, prediction)

    # Return the generated PDF as a downloadable file
    return send_file(report_path, as_attachment=True)

# To run the app
if __name__ == '__main__':
    # Make sure templates folder exists and set the right working directory
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True)
