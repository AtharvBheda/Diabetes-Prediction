import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiabetesModelTrainer:
    def __init__(self, data_path='diabetes.csv'):
        self.data_path = data_path
        self.features = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']
        self.model = None
        self.scaler = None
        self.median_values = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        logger.info("Loading dataset...")
        self.dataset = pd.read_csv(self.data_path)
        logger.info(f"Dataset loaded with {len(self.dataset)} rows and {len(self.dataset.columns)} columns")
        
        # Calculate median values for zero replacement
        self.median_values = {}
        for feature in self.features:
            self.median_values[feature] = self.dataset[feature].median()
            
        logger.info(f"Median values calculated: {self.median_values}")
        
    def preprocess_data(self):
        """Preprocess the data for training"""
        logger.info("Preprocessing data...")
        
        # Separate features and target
        X = self.dataset[self.features]
        y = self.dataset['Outcome']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the SVM model"""
        logger.info("Training SVM model...")
        
        # Try different kernels and select best one
        kernels = ['linear', 'rbf', 'poly']
        best_model = None
        best_score = 0
        
        for kernel in kernels:
            model = svm.SVC(kernel=kernel, probability=True, random_state=42)
            model.fit(X_train, y_train)
            
            # Cross-validation score would be better, but for simplicity using training score
            score = model.score(X_train, y_train)
            logger.info(f"Kernel {kernel}: Training score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        self.model = best_model
        logger.info(f"Best model selected with score: {best_score:.4f}")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        logger.info("Evaluating model...")
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        return accuracy
    
    def save_model(self, model_dir='models'):
        """Save the trained model and preprocessing objects"""
        logger.info("Saving model...")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'diabetes_model.pkl')
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Save median values
        median_path = os.path.join(model_dir, 'median_values.pkl')
        joblib.dump(self.median_values, median_path)
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'features': self.features,
            'model_type': str(type(self.model).__name__),
            'kernel': self.model.kernel if hasattr(self.model, 'kernel') else 'unknown'
        }
        
        metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Median values saved to {median_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
    def train_and_save(self):
        """Complete training pipeline"""
        logger.info("Starting model training pipeline...")
        
        self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data()
        self.train_model(X_train, y_train)
        accuracy = self.evaluate_model(X_test, y_test)
        self.save_model()
        
        logger.info(f"Training completed successfully with accuracy: {accuracy:.4f}")
        return accuracy

def main():
    """Main function to train and save the model"""
    trainer = DiabetesModelTrainer()
    trainer.train_and_save()

if __name__ == '__main__':
    main()
