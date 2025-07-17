# Diabetes Prediction App

A simple Flask web application that predicts diabetes risk using machine learning.

## Features

- **Machine Learning Model**: Uses SVM to predict diabetes based on health parameters
- **Web Interface**: Simple form for inputting health data
- **PDF Report**: Generates a PDF report with prediction results and data visualizations
- **Risk Assessment**: Visual risk level chart for each health parameter
- **Health Parameter Validation**: Validates input data ranges

## Technologies Used

- **Flask**: Web framework
- **scikit-learn**: Machine learning model
- **matplotlib**: Data visualization
- **pdfkit**: PDF generation
- **pandas**: Data processing

## Prerequisites

- Python 3.7 or higher
- pip package manager
- wkhtmltopdf (for PDF generation)

### Installing wkhtmltopdf

**Windows:**
- Download from: https://wkhtmltopdf.org/downloads.html
- Install and add to PATH

**macOS:**
```bash
brew install wkhtmltopdf
```

**Ubuntu/Debian:**
```bash
sudo apt-get install wkhtmltopdf
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AtharvBheda/Diabetes-Prediction.git
   cd Diabetes-Prediction
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (first time only):
   ```bash
   python model_trainer.py
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Open your browser** and go to:
   ```
   http://127.0.0.1:5000
   ```

## Usage

1. Fill in the health parameters in the form:
   - Number of pregnancies
   - Glucose level (mg/dL)
   - Blood pressure (mmHg)
   - BMI (kg/m²)
   - Age (years)
   - Insulin level (optional)

2. Click "Generate Assessment Report"

3. Download the generated PDF report with your diabetes risk assessment

## Project Structure

```
Diabetes-Prediction/
├── app.py              # Main Flask application
├── model_trainer.py    # Model training script
├── model_predictor.py  # Prediction logic
├── report_generator.py # PDF report generation
├── validators.py       # Input validation
├── config.py          # Configuration
├── test_app.py        # Simple tests
├── diabetes.csv       # Training data
├── requirements.txt   # Dependencies
├── models/            # Trained model files
├── templates/         # HTML templates
└── static/           # CSS files
```

## Health Parameter Ranges

| Parameter | Range | Normal Range |
|-----------|-------|--------------|
| Glucose | 0-300 mg/dL | 70-100 mg/dL |
| Blood Pressure | 0-200 mmHg | 60-90 mmHg |
| BMI | 10-70 kg/m² | 18.5-25 kg/m² |
| Age | 10-120 years | - |
| Pregnancies | 0-20 | - |
| Insulin | 0-900 pmol/L | 16-166 pmol/L |

## Testing

Run the test suite:
```bash
python test_app.py
```

## API Endpoints

- `GET /` - Display the main form
- `POST /` - Submit form data and get PDF report
- `GET /health` - Health check endpoint

## Model Information

- **Algorithm**: Support Vector Machine (SVM)
- **Dataset**: Pima Indians Diabetes Database
- **Features**: 6 health parameters
- **Accuracy**: ~77% (varies based on training)

## Screenshots

### Main Form
The application provides a clean, user-friendly interface for inputting health parameters.

### PDF Report
Generated reports include:
- Prediction result with probability
- Risk factor assessment chart
- Data visualization comparing user values to population
- Health parameter ranges and recommendations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Important Notes

- This is for educational purposes only
- Always consult healthcare professionals for medical advice
- The model uses the Pima Indians Diabetes Dataset
- Results should not be used for actual medical diagnosis

## License

This project is for educational use. Feel free to modify and distribute for learning purposes.

## Support

If you encounter any issues:
1. Check that all dependencies are installed
2. Ensure wkhtmltopdf is properly installed
3. Run the test suite to identify problems
4. Create an issue on GitHub with error details

---

**Disclaimer**: This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
