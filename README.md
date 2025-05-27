# Massive Churn Predictor 

A full-featured machine learning pipeline that predicts customer churn using real-world telecom data. Built with Python and scikit-learn, this project demonstrates a complete workflow from data preprocessing to model evaluation and reporting.

## Features

- Cleans and preprocesses real-world customer data
- Encodes categorical variables and scales features
- Trains a Random Forest Classifier to predict churn
- Evaluates the model with accuracy, classification report, and confusion matrix
- Saves the trained model as a `.pkl` file
- Exports test set predictions to a `.csv` report

## How to Run

1. **Install dependencies**

   pip install -r requirements.txt

2. **Run the ML pipeline**

   python churn_pipeline.py

   This will:
   - Train the model
   - Show a confusion matrix chart
   - Save a model file (`churn_model.pkl`)
   - Export a prediction report (`report_output.csv`)

## Project Files

- churn_pipeline.py — Main script that runs the entire pipeline
- customer_churn.csv — Input dataset used for training/testing
- churn_model.pkl — Trained model saved for reuse
- report_output.csv — CSV file with actual vs. predicted churn
- requirements.txt — List of required Python packages

## Example Outputs

- Accuracy: ~80%
- Confusion Matrix: Visual representation of true/false predictions
- Classification Report: Includes precision, recall, and F1-score

## Built With

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
