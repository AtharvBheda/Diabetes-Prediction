import matplotlib.pyplot as plt
import pandas as pd
import pdfkit
import base64
from io import BytesIO
import os
from datetime import datetime

class ReportGenerator:
    def __init__(self, dataset_path='diabetes.csv'):
        self.dataset = pd.read_csv(dataset_path)
        self.features = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']
        
    def generate_simple_charts(self, input_data):
        """Generate simple histograms for key features"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Key features to plot
        key_features = ['Glucose', 'BMI', 'Age', 'BloodPressure']
        
        for i, feature in enumerate(key_features):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            user_value = input_data.get(feature, 0)
            
            # Create histogram
            ax.hist(self.dataset[feature], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            ax.axvline(user_value, color='red', linestyle='--', linewidth=2, label=f'Your Value: {user_value}')
            
            # Add mean line
            mean_val = self.dataset[feature].mean()
            ax.axvline(mean_val, color='green', linestyle=':', linewidth=2, label=f'Average: {mean_val:.1f}')
            
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        image_io = BytesIO()
        plt.savefig(image_io, format='png', dpi=150, bbox_inches='tight')
        image_io.seek(0)
        image_base64 = base64.b64encode(image_io.getvalue()).decode()
        
        plt.close()
        
        return image_base64
    
    def generate_risk_factors_chart(self, input_data):
        """Generate a risk factors visualization"""
        # Define risk thresholds
        risk_thresholds = {
            'Glucose': {'low': 100, 'high': 140},
            'BMI': {'low': 25, 'high': 35},
            'Age': {'low': 45, 'high': 65},
            'BloodPressure': {'low': 90, 'high': 140},
            'Insulin': {'low': 100, 'high': 300},
            'Pregnancies': {'low': 3, 'high': 8}
        }
        
        # Calculate risk scores
        risk_scores = []
        feature_names = []
        
        for feature, value in input_data.items():
            if feature in risk_thresholds:
                thresholds = risk_thresholds[feature]
                
                if value <= thresholds['low']:
                    risk_score = 1  # Low risk
                elif value <= thresholds['high']:
                    risk_score = 2  # Medium risk
                else:
                    risk_score = 3  # High risk
                
                risk_scores.append(risk_score)
                feature_names.append(feature)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if score == 1 else 'orange' if score == 2 else 'red' 
                 for score in risk_scores]
        
        bars = ax.barh(feature_names, risk_scores, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, feature) in enumerate(zip(bars, feature_names)):
            value = input_data[feature]
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{value}', va='center', fontweight='bold')
        
        ax.set_xlabel('Risk Level', fontsize=12)
        ax.set_title('Individual Risk Factors Assessment', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 4)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Low', 'Medium', 'High'])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Low Risk'),
            Patch(facecolor='orange', alpha=0.7, label='Medium Risk'),
            Patch(facecolor='red', alpha=0.7, label='High Risk')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Convert to base64
        image_io = BytesIO()
        plt.savefig(image_io, format='png', dpi=150, bbox_inches='tight')
        image_io.seek(0)
        image_base64 = base64.b64encode(image_io.getvalue()).decode()
        
        plt.close()
        
        return image_base64
    
    def generate_report(self, input_data, prediction, probability, output_path=None):
        """Generate a simple PDF report"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'diabetes_report_{timestamp}.pdf'
        
        # Generate charts
        charts = self.generate_simple_charts(input_data)
        risk_chart = self.generate_risk_factors_chart(input_data)
        
        # Create simple HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; color: #2c3e50; margin-bottom: 30px; }}
                .result {{ 
                    background: {'#ffebee' if prediction == 1 else '#e8f5e8'};
                    border: 2px solid {'#f44336' if prediction == 1 else '#4caf50'};
                    padding: 20px; margin: 20px 0; text-align: center; border-radius: 10px;
                }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                .disclaimer {{ font-size: 12px; color: #666; margin-top: 30px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Diabetes Prediction Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="result">
                <h2>Prediction Result: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}</h2>
                <p>Probability: {probability * 100:.1f}%</p>
            </div>
            
            <h3>Your Input Data</h3>
            <table>
                <tr><th>Parameter</th><th>Your Value</th><th>Normal Range</th></tr>
                <tr><td>Pregnancies</td><td>{input_data['Pregnancies']}</td><td>Varies</td></tr>
                <tr><td>Glucose</td><td>{input_data['Glucose']}</td><td>70-100 mg/dL</td></tr>
                <tr><td>Blood Pressure</td><td>{input_data['BloodPressure']}</td><td>60-90 mmHg</td></tr>
                <tr><td>Insulin</td><td>{input_data['Insulin']}</td><td>16-166 pmol/L</td></tr>
                <tr><td>BMI</td><td>{input_data['BMI']}</td><td>18.5-25 kg/m²</td></tr>
                <tr><td>Age</td><td>{input_data['Age']}</td><td>-</td></tr>
            </table>
            
            <div class="chart">
                <h3>Risk Factors Assessment</h3>
                <img src="data:image/png;base64,{risk_chart}" style="max-width: 100%;">
            </div>
            
            <div class="chart">
                <h3>Data Visualization</h3>
                <img src="data:image/png;base64,{charts}" style="max-width: 100%;">
            </div>
            
            <div class="disclaimer">
                <p><strong>Disclaimer:</strong> This report is for educational purposes only. 
                Please consult with a healthcare provider for proper medical advice.</p>
            </div>
        </body>
        </html>
        """
        
        # Generate PDF
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None
        }
        
        pdfkit.from_string(html_template, output_path, options=options)
        
        return output_path
