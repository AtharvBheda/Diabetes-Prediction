import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pdfkit
from jinja2 import Template
import base64
from io import BytesIO

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Define the columns used for prediction
columns_for_prediction = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']

# Separate the data and labels
X = diabetes_dataset[columns_for_prediction]
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_Train, Y_Train)

# Function to replace 0 values with the median value from the dataset
def replace_zero_with_median(input_data):
    input_df = pd.DataFrame([input_data], columns=columns_for_prediction)
    for column in columns_for_prediction:
        if input_df[column].values[0] == 0:
            median_value = diabetes_dataset[column].median()
            input_df[column] = median_value
    return input_df.values.flatten()

# Function to make predictions
def predict_diabetes(input_data):
    """
    Takes unscaled input_data as a tuple, standardizes it, and returns the prediction result.
    """
    # Replace 0 values with median values
    input_data_cleaned = replace_zero_with_median(input_data)
    
    # Standardize the input data
    std_data = scaler.transform([input_data_cleaned])
    
    # Make the prediction using the trained classifier
    prediction = classifier.predict(std_data)
    
    return prediction

# Function to generate histograms and the report
def generate_report(input_data, prediction, output_path='diabetes_report.pdf'):
    # Load the template
    with open('templates\\report_template.html') as file:
        template = Template(file.read())

    # Prepare data for the template
    prediction_text = "The person is Diabetic." if prediction[0] == 1 else "The person is Non-Diabetic."
    
    input_data_dict = {feature: value for feature, value in zip(columns_for_prediction, input_data)}

    # Generate histograms for selected features and encode them in base64
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # Adjust layout (2 rows, 2 columns)
    features_to_plot = ['Glucose', 'BMI', 'Age', 'BloodPressure']

    # Now map the correct input values to the corresponding features
    user_values_for_plotting = [
        input_data_dict['Glucose'],
        input_data_dict['BMI'],
        input_data_dict['Age'],
        input_data_dict['BloodPressure']
    ]

    # Plot raw values (unscaled) for the red lines
    for ax, feature, user_value in zip(axes.flatten(), features_to_plot, user_values_for_plotting):
        generate_feature_histogram(feature, user_value, diabetes_dataset, ax)

    plt.tight_layout()  # Ensure everything fits within the figure
    image_io = BytesIO()
    plt.savefig(image_io, format='png')
    image_io.seek(0)
    image_base64 = base64.b64encode(image_io.getvalue()).decode()

    # Render the template
    html = template.render(prediction_text=prediction_text, input_data=input_data_dict, image1=image_base64)
    
    # Generate the PDF
    pdfkit.from_string(html, output_path)

    # Return the path to the generated PDF
    return output_path

# Function to generate histograms for the input data (this remains the same)
def generate_feature_histogram(feature_name, user_value, dataset, ax):
    ax.hist(dataset[feature_name], bins=20, color='lightblue', edgecolor='black')
    ax.axvline(user_value, color='red', linestyle='dashed', linewidth=2)  # Mark the raw value (unscaled)
    ax.set_title(f'{feature_name} Distribution')
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Frequency')
